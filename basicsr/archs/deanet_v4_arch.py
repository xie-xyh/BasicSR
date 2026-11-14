# DEANetV4: frozen 004a backbone (DEANetV3-LFM) + SDA-style haze-density adapter.
#
# 006a probe (see research/006_hihaze/README.md): ports the Spatial Degradation
# Adapter idea from Hihaze (Chen et al., 2026, arXiv:2610.25832) to the
# deterministic DEA-Net family. A lightweight conv encoder extracts multi-scale
# degradation (haze-density) features from the hazy input itself; the features
# are injected into the decoder through zero-initialized adapters, so the whole
# network starts exactly identical to the frozen 004a backbone.
#
# Injection points (P1/P2 from the 006a design doc; P3 LFM-coupling is phase 2):
#   P1: after mix2 (level-2 decoder entry, 1/2 res)  — haze-density guides
#       where detail reconstruction should be strong vs conservative.
#   P2: before up3 (level-1 decoder tail, full res)   — negative gating against
#       over-sharpening thin-haze / noisy regions.
#   OUT: zero-init residual correction on the 3ch output (Hihaze's noise
#       correction branch analogue).
# The coarsest (1/4) SDA scale is supervised but not injected in 006a
# (reserved for the phase-2 LFM coupling).
#
# SDA supervision heads predict (T_d, T_r, T_s) maps at each scale; the targets
# are built from Sobel/contrast statistics in SDAModel (paired GT required).

import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v3_arch import DEANetV3


class SDAEncoder(nn.Module):
    """Multi-scale degradation-feature encoder (Hihaze SDA encoder analogue).

    Args:
        in_ch (int): Input channels. Default: 3.
        dims (tuple): Channel numbers at (full, 1/2, 1/4) resolutions.
    """

    def __init__(self, in_ch=3, dims=(16, 32, 64)):
        super(SDAEncoder, self).__init__()
        d1, d2, d3 = dims
        self.dims = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, d1, 3, padding=1), nn.GELU(),
            nn.Conv2d(d1, d1, 3, padding=1), nn.GELU())
        self.down1 = nn.Sequential(
            nn.Conv2d(d1, d2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d2, d2, 3, padding=1), nn.GELU())
        self.down2 = nn.Sequential(
            nn.Conv2d(d2, d3, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(d3, d3, 3, padding=1), nn.GELU())
        # per-scale supervision heads -> (T_d, T_r, T_s) maps
        self.head1 = nn.Conv2d(d1, 3, 1)
        self.head2 = nn.Conv2d(d2, 3, 1)
        self.head3 = nn.Conv2d(d3, 3, 1)

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f4 = self.down2(f2)
        preds = (self.head1(f1), self.head2(f2), self.head3(f4))
        return (f1, f2, f4), preds


@ARCH_REGISTRY.register()
class DEANetV4(DEANetV3):
    """004a backbone (frozen) + SDA degradation-density adapters.

    Args:
        backbone_ckpt (str): Path to the 004a checkpoint (params_ema preferred).
        base_dim (int): Channel number of the first level. Default: 32.
        in_ch (int): Number of input channels. Default: 3.
        out_ch (int): Number of output channels. Default: 3.
        global_block (str): Bottleneck block of the backbone. Default: 'lfm'.
        sda_dims (tuple): SDA encoder channels at (full, 1/2, 1/4) resolutions.
    """

    def __init__(self, backbone_ckpt, base_dim=32, in_ch=3, out_ch=3,
                 global_block='lfm', sda_dims=(16, 32, 64)):
        super(DEANetV4, self).__init__(base_dim, in_ch, out_ch, global_block)
        sd = torch.load(backbone_ckpt, map_location='cpu')
        key = 'params_ema' if 'params_ema' in sd else 'params'
        self.load_state_dict(sd[key], strict=True)
        self.requires_grad_(False)

        self.sda_encoder = SDAEncoder(in_ch, sda_dims)
        d1, d2, _ = sda_dims
        self.adapter_mix2 = nn.Conv2d(base_dim * 2 + d2, base_dim * 2, 1)
        self.adapter_up3 = nn.Conv2d(base_dim + d1, base_dim, 1)
        self.adapter_out = nn.Conv2d(out_ch + d1, out_ch, 1)
        # zero-init all injection points -> initial output equals 004a exactly
        for m in (self.adapter_mix2, self.adapter_up3, self.adapter_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        (f1, f2, _), preds = self.sda_encoder(x)

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
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x2 = self.level3_fgb1(x2)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x5 = self.level3_fgb2(x5)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        # P1: density-guided correction at the level-2 decoder entry
        x_level2_mix = x_level2_mix + self.adapter_mix2(torch.cat([x_level2_mix, f2], dim=1))

        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        # P2: density-guided correction before the output head
        x_up2 = x_up2 + self.adapter_up3(torch.cat([x_up2, f1], dim=1))

        out = self.up3(x_up2)
        # OUT: zero-init output residual correction
        out = out + self.adapter_out(torch.cat([out, f1], dim=1))

        if self.training:
            self.last_sda_preds = preds
            with torch.no_grad():
                self.last_base = super(DEANetV4, self).forward(x)
        return out
