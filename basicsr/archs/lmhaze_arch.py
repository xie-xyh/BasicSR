import math
from functools import partial
import torch
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch import nn
from torch.nn import functional as F

from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None


def selective_scan_ref(u,
                       delta,
                       A,
                       B,
                       C,
                       D=None,
                       z=None,
                       delta_bias=None,
                       delta_softplus=False,
                       return_last_state=False):
    """Pure PyTorch reference for environments without the mamba-ssm CUDA extension."""
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, length = u.shape
    state_dim = A.shape[1]
    if B.ndim == 4:
        groups = B.shape[1]
        B = B.unsqueeze(2).expand(batch, groups, dim // groups, state_dim,
                                  length).reshape(batch, dim, state_dim, length)
        C = C.unsqueeze(2).expand(batch, groups, dim // groups, state_dim,
                                  length).reshape(batch, dim, state_dim, length)
        delta_B_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    else:
        delta_B_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    delta_A = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    state = torch.zeros((batch, dim, state_dim), device=u.device, dtype=torch.float32)
    outputs = []
    for i in range(length):
        state = delta_A[:, :, i, :] * state + delta_B_u[:, :, i, :]
        if C.ndim == 4:
            outputs.append(torch.einsum('bdn,bdn->bd', state, C[:, :, :, i]))
        else:
            outputs.append(torch.einsum('bdn,bn->bd', state, C[:, :, i]))
    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + u * D.reshape(1, -1, 1)
    if z is not None:
        output = output * F.silu(z)
    if return_last_state:
        return output.to(dtype_in), state
    return output.to(dtype_in)


class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1), nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1), nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1), ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.cab(x)


class SSMBlock(nn.Module):

    def __init__(self,
                 d_model,
                 d_state=16,
                 d_conv=3,
                 expand=2,
                 dt_rank='auto',
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init='random',
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 dropout=0.,
                 conv_bias=True,
                 bias=False,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs)
        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs) for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))
        del x_proj

        dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))
        del dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn or selective_scan_ref
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(dt_rank,
                d_inner,
                dt_scale=1.0,
                dt_init='random',
                dt_min=0.001,
                dt_max=0.1,
                dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == 'constant':
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == 'random':
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f'Unsupported dt initialization: {dt_init}.')

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) +
            math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), 'n -> d n', d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, 'd n -> r d n', r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, 'n1 -> r n1', r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        batch, _, height, width = x.shape
        length = height * width
        x_hwwh = torch.stack([x.reshape(batch, -1, length),
                              x.transpose(2, 3).contiguous().reshape(batch, -1, length)],
                             dim=1)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum('b k d l, k c d -> b k c l', xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l', dts, self.dt_projs_weight)

        xs = xs.float().reshape(batch, -1, length)
        dts = dts.contiguous().float().reshape(batch, -1, length)
        Bs = Bs.float().reshape(batch, 4, -1, length)
        Cs = Cs.float().reshape(batch, 4, -1, length)
        Ds = self.Ds.float().reshape(-1)
        As = -torch.exp(self.A_logs.float()).reshape(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().reshape(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True,
            return_last_state=False).reshape(batch, 4, -1, length)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1])
        wh_y = out_y[:, 1].reshape(batch, -1, width, height).transpose(2, 3).contiguous().reshape(batch, -1, length)
        invwh_y = inv_y[:, 1].reshape(batch, -1, width, height).transpose(2, 3).contiguous().reshape(batch, -1, length)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, batch, height, width):
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = y.transpose(1, 2).contiguous().reshape(batch, height, width, -1)
        return self.out_norm(y)


class VSSMMoE(nn.Module):

    def __init__(self,
                 d_model,
                 num_experts=14,
                 expert_groups=((0, 5, 3), (5, 8, 2), (8, 11, 1), (11, 14, 1)),
                 expand=2,
                 dropout=0.,
                 bias=False,
                 device=None,
                 dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_inner = int(expand * d_model)
        self.num_experts = num_experts
        self.expert_groups = tuple(tuple(group) for group in expert_groups)
        if sorted(index for start, end, _ in self.expert_groups
                  for index in range(start, end)) != list(range(num_experts)):
            raise ValueError('expert_groups must cover every expert exactly once and in order.')
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.moe_blocks = nn.ModuleList([SSMBlock(d_model=d_model) for _ in range(num_experts)])
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, logit, use_gates=True):
        batch, height, width, _ = x.shape
        if logit.ndim != 2 or logit.shape != (batch, self.num_experts):
            raise ValueError(
                f'Expected routing logits with shape ({batch}, {self.num_experts}), got {tuple(logit.shape)}.')
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        logit = logit.float()

        topk_results = [logit[:, start:end].topk(topk, dim=1) for start, end, topk in self.expert_groups]
        selected_weights = torch.softmax(torch.cat([result.values for result in topk_results], dim=1), dim=1)
        selected_indices = torch.cat(
            [result.indices + group[0] for result, group in zip(topk_results, self.expert_groups)], dim=1)
        if use_gates:
            gates = logit.new_zeros((batch, self.num_experts)).scatter(1, selected_indices, selected_weights)
        else:
            gates = logit.new_ones((batch, self.num_experts))

        output = x.new_zeros((batch, height, width, self.d_inner))
        for expert_idx, expert in enumerate(self.moe_blocks):
            sample_indices = (selected_indices == expert_idx).any(dim=1).nonzero(as_tuple=False).flatten()
            if sample_indices.numel() == 0:
                continue
            expert_output = expert(x[sample_indices], sample_indices.numel(), height, width)
            expert_gates = gates[sample_indices, expert_idx].reshape(-1, 1, 1, 1)
            output.index_add_(0, sample_indices, expert_output * expert_gates)

        output = self.out_proj(output * F.silu(z))
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class SS2D(SSMBlock):

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0., **kwargs):
        super().__init__(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout, **kwargs)

    def forward(self, x):
        batch, height, width, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = (y1 + y2 + y3 + y4).transpose(1, 2).contiguous().reshape(batch, height, width, -1)
        y = self.out_norm(y) * F.silu(z)
        output = self.out_proj(y)
        if self.dropout is not None:
            output = self.dropout(output)
        return output


class VSSBlock(nn.Module):

    def __init__(self,
                 hidden_dim,
                 drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0.,
                 d_state=16,
                 expand=2.,
                 **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, expand=expand, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, x_size):
        batch, _, channels = x.shape
        x = x.reshape(batch, *x_size, channels).contiguous()
        residual = x
        x = residual * self.skip_scale + self.drop_path(self.self_attention(self.ln_1(x)))
        conv = self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + conv
        return x.reshape(batch, -1, channels).contiguous()


class VSSBlockMoE(nn.Module):

    def __init__(self,
                 hidden_dim,
                 num_experts=14,
                 expert_groups=((0, 5, 3), (5, 8, 2), (8, 11, 1), (11, 14, 1)),
                 drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate=0.,
                 expand=2.,
                 **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VSSMMoE(
            d_model=hidden_dim,
            num_experts=num_experts,
            expert_groups=expert_groups,
            expand=expand,
            dropout=attn_drop_rate,
            bias=kwargs.get('bias', False))
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, logit, x_size, use_gates=True):
        batch, _, channels = x.shape
        x = x.reshape(batch, *x_size, channels).contiguous()
        residual = x
        x = residual * self.skip_scale + self.drop_path(self.self_attention(self.ln_1(x), logit, use_gates))
        conv = self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x * self.skip_scale2 + conv
        return x.reshape(batch, -1, channels).contiguous()


class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1, bias=bias)

    def forward(self, x):
        return rearrange(self.proj(x), 'b c h w -> b (h w) c').contiguous()


class Downsample(nn.Module):

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))

    def forward(self, x, height, width):
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width).contiguous()
        return rearrange(self.body(x), 'b c h w -> b (h w) c').contiguous()


class Upsample(nn.Module):

    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x, height, width):
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width).contiguous()
        return rearrange(self.body(x), 'b c h w -> b (h w) c').contiguous()


class IntensityRouter(nn.Module):

    def __init__(self, inp_channels=3, num_experts=14, hidden_dim=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(inp_channels * 2, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, num_experts))

    def forward(self, x):
        mean = self.pool(x)
        second_moment = self.pool(x * x)
        return self.mlp(torch.cat([mean, second_moment], dim=1))


@ARCH_REGISTRY.register()
class LMHaze(nn.Module):
    """MoE-Mamba U-Net from LMHaze with a self-contained intensity router.

    The official network is MambaIRUNet in LMHaze's Dehazing directory. This
    implementation keeps its four-direction SSM blocks, 14 grouped experts,
    Top-K routing and U-Net topology. External logits retain the original
    equally weighted expert sum. When logits are absent, image statistics drive
    a learned router with normalized gates so that it receives gradients.
    """

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=(4, 6, 6, 8),
                 mlp_ratio=2.,
                 num_refinement_blocks=4,
                 drop_path_rate=0.,
                 bias=False,
                 dual_pixel_task=False,
                 num_experts=14,
                 expert_groups=((0, 5, 3), (5, 8, 2), (8, 11, 1), (11, 14, 1)),
                 router_dim=64,
                 use_internal_router=True):
        super().__init__()
        if selective_scan_fn is None:
            get_root_logger().warning('mamba-ssm is unavailable; LMHaze uses the slow pure PyTorch selective scan.')
        if len(num_blocks) != 4:
            raise ValueError('num_blocks must contain four stage depths.')
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim, bias)
        self.router = IntensityRouter(inp_channels, num_experts, router_dim) if use_internal_router else None
        base_d_state = 4
        block_kwargs = dict(
            num_experts=num_experts,
            expert_groups=expert_groups,
            drop_path=drop_path_rate,
            norm_layer=nn.LayerNorm,
            attn_drop_rate=0,
            expand=mlp_ratio,
            bias=bias)

        self.encoder_level1_moe = nn.ModuleList([VSSBlockMoE(hidden_dim=dim, d_state=base_d_state, **block_kwargs)])
        self.encoder_level1 = nn.ModuleList([
            VSSBlock(hidden_dim=dim, d_state=base_d_state, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[0] - 1)
        ])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 2, d_state=base_d_state * 2, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 4, d_state=base_d_state * 4, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 8, d_state=base_d_state * 8, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 4, d_state=base_d_state * 4, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[2])
        ])
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 2, d_state=base_d_state * 2, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[1])
        ])
        self.up2_1 = Upsample(dim * 2)
        self.decoder_level1_moe = nn.ModuleList(
            [VSSBlockMoE(hidden_dim=dim * 2, d_state=base_d_state * 2, **block_kwargs)])
        self.decoder_level1 = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 2, d_state=base_d_state * 2, **self._block_kwargs(block_kwargs))
            for _ in range(num_blocks[0] - 1)
        ])
        self.refinement = nn.ModuleList([
            VSSBlock(hidden_dim=dim * 2, d_state=base_d_state * 2, **self._block_kwargs(block_kwargs))
            for _ in range(num_refinement_blocks)
        ])

        self.dual_pixel_task = dual_pixel_task
        if dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, dim * 2, 1, bias=bias)
        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    @staticmethod
    def _block_kwargs(kwargs):
        return {key: value for key, value in kwargs.items() if key not in ('num_experts', 'expert_groups')}

    def forward(self, inp_img, logit=None):
        if inp_img.ndim != 4:
            raise ValueError(f'Expected a BCHW image tensor, got {tuple(inp_img.shape)}.')
        _, _, height, width = inp_img.shape
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f'LMHaze input height and width must be divisible by 8, got {height}x{width}.')
        use_gates = logit is None
        if use_gates:
            if self.router is None:
                raise ValueError('LMHaze requires external logits when use_internal_router is false.')
            logit = self.router(inp_img)
        elif logit.ndim == 1:
            logit = logit.unsqueeze(0)

        out_enc_level1 = self.patch_embed(inp_img)
        for layer in self.encoder_level1_moe:
            out_enc_level1 = layer(out_enc_level1, logit, (height, width), use_gates)
        for layer in self.encoder_level1:
            out_enc_level1 = layer(out_enc_level1, (height, width))

        out_enc_level2 = self.down1_2(out_enc_level1, height, width)
        for layer in self.encoder_level2:
            out_enc_level2 = layer(out_enc_level2, (height // 2, width // 2))

        out_enc_level3 = self.down2_3(out_enc_level2, height // 2, width // 2)
        for layer in self.encoder_level3:
            out_enc_level3 = layer(out_enc_level3, (height // 4, width // 4))

        latent = self.down3_4(out_enc_level3, height // 4, width // 4)
        for layer in self.latent:
            latent = layer(latent, (height // 8, width // 8))

        out_dec_level3 = self.up4_3(latent, height // 8, width // 8)
        out_dec_level3 = torch.cat([out_dec_level3, out_enc_level3], dim=2)
        out_dec_level3 = rearrange(out_dec_level3, 'b (h w) c -> b c h w', h=height // 4, w=width // 4)
        out_dec_level3 = self.reduce_chan_level3(out_dec_level3)
        out_dec_level3 = rearrange(out_dec_level3, 'b c h w -> b (h w) c').contiguous()
        for layer in self.decoder_level3:
            out_dec_level3 = layer(out_dec_level3, (height // 4, width // 4))

        out_dec_level2 = self.up3_2(out_dec_level3, height // 4, width // 4)
        out_dec_level2 = torch.cat([out_dec_level2, out_enc_level2], dim=2)
        out_dec_level2 = rearrange(out_dec_level2, 'b (h w) c -> b c h w', h=height // 2, w=width // 2)
        out_dec_level2 = self.reduce_chan_level2(out_dec_level2)
        out_dec_level2 = rearrange(out_dec_level2, 'b c h w -> b (h w) c').contiguous()
        for layer in self.decoder_level2:
            out_dec_level2 = layer(out_dec_level2, (height // 2, width // 2))

        out_dec_level1 = self.up2_1(out_dec_level2, height // 2, width // 2)
        out_dec_level1 = torch.cat([out_dec_level1, out_enc_level1], dim=2)
        for layer in self.decoder_level1_moe:
            out_dec_level1 = layer(out_dec_level1, logit, (height, width), use_gates)
        for layer in self.decoder_level1:
            out_dec_level1 = layer(out_dec_level1, (height, width))
        for layer in self.refinement:
            out_dec_level1 = layer(out_dec_level1, (height, width))
        out_dec_level1 = rearrange(out_dec_level1, 'b (h w) c -> b c h w', h=height, w=width)

        if self.dual_pixel_task:
            return self.output(out_dec_level1 + self.skip_conv(self.patch_embed.proj(inp_img)))
        return self.output(out_dec_level1) + inp_img
