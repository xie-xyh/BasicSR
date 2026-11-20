# DEANetV8: adaptive-boundary low-frequency parallel pathway with skip gating (012b).
#
# Motivation (research/012_deanetv8/README.md): 004a proved a global
# low-frequency pathway helps (LFM, +0.13 dB), but it is a thin FiLM modulator;
# 012a showed a fixed-boundary capacity branch only matches it (~+0.1 dB,
# below the two-seed threshold). V8 upgrades the pathway with:
#   - content-adaptive frequency boundary fc, clamped to the physically
#     motivated range [0.02, 0.15] Nyquist (residual-spectrum evidence);
#   - one low band -> coarse grid (32x32, where 3x3 convs have a global
#     receptive field by construction) veil-field branch with real capacity;
#   - multi-point zero-init injections (level-3 slots + level-2 decoder);
#   - skip gating at the two CGA fusion points: the veil field suppresses the
#     residual haze carried by shallow skip features, in exp(-s) form so the
#     gate starts exactly at identity (011n: skips are load-bearing).

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v3_arch import DEANetV3


class SGResBlock(nn.Module):
    """Residual block with a NAFNet-style SimpleGate nonlinearity."""

    def __init__(self, dim):
        super(SGResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        a, b = self.conv1(x).chunk(2, dim=1)
        return x + self.conv2(a * b)


class AdaptiveDualFreqBlock(nn.Module):
    """Content-adaptive low band split + coarse-grid veil-field branch.

    Predicts the split boundary fc from the current features and the input
    image statistics, extracts the low band with a Gaussian low-pass mask,
    processes it at branch_res x branch_res (global receptive field by
    construction) and fuses back through a zero-init 1x1 projection.

    Args:
        dim (int): Channel number.
        fc_min (float): Minimum boundary as fraction of Nyquist. Default: 0.02.
        fc_max (float): Maximum boundary as fraction of Nyquist. Default: 0.15.
        branch_res (int): Spatial size of the coarse processing grid.
        n_blocks (int): Number of SGResBlocks in the branch.
    """

    def __init__(self, dim, fc_min=0.02, fc_max=0.15, branch_res=32, n_blocks=3):
        super(AdaptiveDualFreqBlock, self).__init__()
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.branch_res = branch_res
        # zero-init last layer -> z = 0 -> fc = (fc_min + fc_max) / 2 = 0.085,
        # the measured effective cutoff of the LFM family (prior center)
        self.boundary = nn.Sequential(nn.Linear(dim + 6, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.boundary[-1].weight)
        nn.init.zeros_(self.boundary[-1].bias)
        self.branch = nn.Sequential(*[SGResBlock(dim) for _ in range(n_blocks)])
        self.inject = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.inject.weight)
        nn.init.zeros_(self.inject.bias)

    def forward(self, x, stats, apply_injection=True):
        B, C, H, W = x.shape
        pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
        z = self.boundary(torch.cat([pooled, stats], dim=1))
        fc = self.fc_min + (self.fc_max - self.fc_min) * torch.sigmoid(z)  # B,1

        v = 2 * torch.fft.fftfreq(H, device=x.device)[:, None]
        u = 2 * torch.fft.rfftfreq(W, device=x.device)[None, :]
        d2 = (v ** 2 + u ** 2)[None, None]                          # 1,1,H,W/2+1
        mask = torch.exp(-d2 / (2 * fc.view(B, 1, 1, 1) ** 2))      # B,1,H,W/2+1
        spec = torch.fft.rfft2(x.float(), norm='ortho')
        low = torch.fft.irfft2(spec * mask, s=(H, W), norm='ortho').to(x.dtype)

        veil = F.adaptive_avg_pool2d(low, self.branch_res)
        veil = self.branch(veil)
        veil = F.interpolate(veil, size=(H, W), mode='bilinear', align_corners=False)
        if apply_injection:
            x = x + self.inject(veil)
        return x, veil, fc


@ARCH_REGISTRY.register()
class DEANetV8(DEANetV3):
    """DEA-Net with an adaptive-boundary low-frequency pathway and skip gating.

    Same skeleton as 004a except the two LFM slots (after level-3 block2 and
    block5) become AdaptiveDualFreqBlocks; the deeper block's veil field is
    additionally injected into the level-2 decoder mix and used to gate the
    skip features at the CGA fusion points.

    Args:
        base_dim (int): Channel number of the first level. Default: 32.
        in_ch (int): Number of input channels. Default: 3.
        out_ch (int): Number of output channels. Default: 3.
        branch_res (int): Coarse grid size of the veil branch. Default: 32.
        n_blocks (int): Number of SGResBlocks in the branch. Default: 3.
        inject_levels (tuple): Injection points, subset of
            ('block2', 'block5', 'mix2'). Default: all three.
        skip_gate_levels (tuple): CGA fusion points whose skip is gated,
            subset of ('mix1', 'mix2'). Default: both.
    """

    def __init__(self, base_dim=32, in_ch=3, out_ch=3, branch_res=32, n_blocks=3,
                 inject_levels=('block2', 'block5', 'mix2'), skip_gate_levels=('mix1', 'mix2')):
        super(DEANetV8, self).__init__(base_dim, in_ch, out_ch, global_block='lfm')
        del self.level3_fgb1, self.level3_fgb2
        dim3, dim2 = base_dim * 4, base_dim * 2
        for level in inject_levels:
            if level not in ('block2', 'block5', 'mix2'):
                raise ValueError(f'Unknown inject level: {level}')
        for level in skip_gate_levels:
            if level not in ('mix1', 'mix2'):
                raise ValueError(f'Unknown skip-gate level: {level}')
        self.inject_levels = tuple(inject_levels)
        self.skip_gate_levels = tuple(skip_gate_levels)

        self.level3_adfb1 = AdaptiveDualFreqBlock(dim3, branch_res=branch_res, n_blocks=n_blocks)
        self.level3_adfb2 = AdaptiveDualFreqBlock(dim3, branch_res=branch_res, n_blocks=n_blocks)
        # level-2 decoder injection (veil projected to dim2)
        self.veil_proj2 = nn.Conv2d(dim3, dim2, 1)
        self.inject2 = nn.Conv2d(dim2, dim2, 1)
        nn.init.zeros_(self.inject2.weight)
        nn.init.zeros_(self.inject2.bias)
        # skip suppression gates, exp(-s) form: zero-init -> exact identity
        self.skip_suppress3 = nn.Conv2d(dim3 * 2, dim3, 3, padding=1)
        self.skip_suppress2 = nn.Conv2d(dim2 * 2, dim2, 3, padding=1)
        nn.init.zeros_(self.skip_suppress3.weight)
        nn.init.zeros_(self.skip_suppress3.bias)
        nn.init.zeros_(self.skip_suppress2.weight)
        nn.init.zeros_(self.skip_suppress2.bias)
        self.diagnostics = {}

    def forward(self, x):
        stats = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)  # B,6

        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)

        x_down3 = self.down3(x_down2_init)
        h = self.fe_level_3(x_down3)
        h = self.level3_block1(h)
        h = self.level3_block2(h)
        h, veil1, fc1 = self.level3_adfb1(h, stats, apply_injection='block2' in self.inject_levels)
        h = self.level3_block3(h)
        h = self.level3_block4(h)
        h = self.level3_block5(h)
        h, veil2, fc2 = self.level3_adfb2(h, stats, apply_injection='block5' in self.inject_levels)
        h = self.level3_block6(h)
        h = self.level3_block7(h)
        h = self.level3_block8(h)

        s3 = None
        if 'mix1' in self.skip_gate_levels:
            s3 = self.skip_suppress3(torch.cat([x_down3, veil2], dim=1))
            x_down3 = x_down3 * torch.exp(-s3)
        x_level3_mix = self.mix1(x_down3, h)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        veil_l2 = self.veil_proj2(F.interpolate(veil2, size=x_down2.shape[-2:], mode='bilinear',
                                                align_corners=False))
        s2 = None
        if 'mix2' in self.skip_gate_levels:
            s2 = self.skip_suppress2(torch.cat([x_down2, veil_l2], dim=1))
            x_down2 = x_down2 * torch.exp(-s2)
        x_level2_mix = self.mix2(x_down2, x_up1)
        if 'mix2' in self.inject_levels:
            x_level2_mix = x_level2_mix + self.inject2(veil_l2)

        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        self.diagnostics = {
            'fc1': fc1.mean().detach(), 'fc2': fc2.mean().detach(),
            'veil1_abs': veil1.abs().mean().detach(), 'veil2_abs': veil2.abs().mean().detach(),
        }
        if s3 is not None:
            self.diagnostics['skip3_s'] = s3.abs().mean().detach()
        if s2 is not None:
            self.diagnostics['skip2_s'] = s2.abs().mean().detach()
        return out
