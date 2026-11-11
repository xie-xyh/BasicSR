# DEANetV3: DEA-Net + content-adaptive global block at the bottleneck.
#
# Follow-up of 003d/DEANetV2 (see research/003d_deanetv2/README.md §7): the static
# GFNet-style filter (FGB) failed — it drifted to mid/high frequencies, and its
# GroupNorm stripped the global statistics it was meant to model. Residual
# spectrum analysis shows 55-61% of the residual energy still lives in the
# DC~0.02 Nyquist band, so the low-frequency global direction stands, but the
# global block must be *content-adaptive*.
#
# Two candidate blocks, selectable via `global_block`:
#   - 'lfm'  (LowFreqModBlock): FFT soft-mask low-band split + FiLM modulation of
#     the low band only, with spatially-varying scale/shift maps predicted from a
#     16x16 downsampled view of the input (haze/tone vary smoothly over space).
#     No normalization; zero-init -> identity. In the spirit of AdaIR (ICLR 2025)
#     frequency mining/modulation, restricted to the low band by construction.
#   - 'mdta' (MDTABlock): Restormer-style channel self-attention + gated FFN,
#     a generic content-adaptive global operator as a non-frequency control.
#
# Both replace the two FreqGlobalBlocks of DEANetV2 at the same positions
# (after level3 block2 and block5), so 003c/003d/004x differ only at the
# bottleneck global module.

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v2_arch import DEANetV2


class LowFreqModBlock(nn.Module):
    """Content-adaptive low-frequency FiLM modulation block.

    x -> rFFT2 -> soft radial mask M -> low band -> FiLM(gamma, beta) -> residual
    (gamma, beta) are predicted from a low-resolution (mod_size) view of x, so the
    correction is content-adaptive and smoothly spatially varying.

    Args:
        dim (int): Channel number.
        cutoff (float): Low-pass cutoff as fraction of Nyquist (smaller dim).
        edge (float): Soft transition width of the mask (same units).
        mod_size (int): Spatial size of the modulation prediction map.
        reduction (int): Channel reduction of the modulation predictor.
    """

    def __init__(self, dim, cutoff=0.08, edge=0.04, mod_size=16, reduction=4):
        super(LowFreqModBlock, self).__init__()
        self.cutoff = cutoff
        self.edge = edge
        self.mod_size = mod_size
        hidden = max(dim // reduction, 16)
        self.mod = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, dim * 2, 1))
        # zero-init the output conv -> gamma=0, beta=0 -> identity at start
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    @staticmethod
    def _soft_lowpass_mask(h, w, cutoff, edge, device, dtype):
        nyq = min(h, w) / 2.0
        v = torch.fft.fftfreq(h, device=device) * h  # 0..h/2 wrapped
        u = torch.fft.rfftfreq(w, device=device) * w
        r = torch.sqrt(v[:, None]**2 + u[None, :]**2) / nyq
        m = torch.sigmoid((cutoff - r) / edge)
        return m.to(dtype)[None, None, :, :]  # 1,1,h,w//2+1

    def forward(self, x):
        B, C, H, W = x.shape
        mask = self._soft_lowpass_mask(H, W, self.cutoff, self.edge, x.device, torch.float32)
        spec = torch.fft.rfft2(x.float(), norm='ortho')
        low = torch.fft.irfft2(spec * mask, s=(H, W), norm='ortho').to(x.dtype)

        g = F.adaptive_avg_pool2d(x, self.mod_size)
        g = self.mod(g)  # B, 2C, mod_size, mod_size
        g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        gamma, beta = g.chunk(2, dim=1)
        return x + gamma * low + beta


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim for NCHW tensors (Restormer-style)."""

    def __init__(self, dim):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + 1e-6)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class MDTABlock(nn.Module):
    """Restormer-style block: channel self-attention (MDTA) + gated FFN (GDFN).

    Content-adaptive global operator: attention is computed across channels with
    per-head learnable temperature; spatial complexity is linear in HW.
    Output projections are zero-init -> identity at start.
    """

    def __init__(self, dim, num_heads=4, ffn_expansion=2, bias=False):
        super(MDTABlock, self).__init__()
        self.num_heads = num_heads
        self.norm1 = LayerNorm2d(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
        self.norm2 = LayerNorm2d(dim)
        hidden = dim * ffn_expansion
        self.ffn_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.ffn_dw = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=bias)
        self.ffn_out = nn.Conv2d(hidden, dim, 1, bias=bias)
        nn.init.zeros_(self.project_out.weight)
        if bias:
            nn.init.zeros_(self.project_out.bias)
        nn.init.zeros_(self.ffn_out.weight)
        if bias:
            nn.init.zeros_(self.ffn_out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv_dw(self.qkv(self.norm1(x))).chunk(3, dim=1)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.temperature, dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        x = x + self.project_out(out)
        a, b = self.ffn_dw(self.ffn_in(self.norm2(x))).chunk(2, dim=1)
        return x + self.ffn_out(F.gelu(a) * b)


@ARCH_REGISTRY.register()
class DEANetV3(DEANetV2):
    """DEA-Net with content-adaptive global blocks at the level-3 bottleneck.

    Same skeleton as DEANet/DEANetV2; the two global blocks after level3
    block2 and block5 are chosen by `global_block`:
        'lfm'  -> LowFreqModBlock (frequency-domain, low-band only)
        'mdta' -> MDTABlock (spatial-domain channel attention control)

    Args:
        base_dim (int): Channel number of the first level. Default: 32.
        in_ch (int): Number of input channels. Default: 3.
        out_ch (int): Number of output channels. Default: 3.
        global_block (str): 'lfm' or 'mdta'. Default: 'lfm'.
    """

    def __init__(self, base_dim=32, in_ch=3, out_ch=3, global_block='lfm'):
        super(DEANetV3, self).__init__(base_dim, in_ch, out_ch)
        dim = base_dim * 4
        if global_block == 'lfm':
            blk1, blk2 = LowFreqModBlock(dim), LowFreqModBlock(dim)
        elif global_block == 'mdta':
            blk1, blk2 = MDTABlock(dim), MDTABlock(dim)
        else:
            raise ValueError(f"unknown global_block: {global_block} (expect 'lfm' or 'mdta')")
        # reuse DEANetV2's insertion points (forward references these names)
        self.level3_fgb1 = blk1
        self.level3_fgb2 = blk2
