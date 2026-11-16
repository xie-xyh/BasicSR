# DEANetV5: DEA-Net + parameterized LFM global-modulation stack (007 series).
#
# Motivation (research/007_rf_capacity/README.md): DEA-Net's bottleneck
# theoretical receptive field is only ~197 px while haze/tone errors span the
# whole image; 004a added a *global* path via two LowFreqModBlocks, but the
# global capacity is thin (2 blocks, 16x16 modulation grid). V5 parameterizes
# the LFM stack so its depth (positions), modulation resolution (mod_size) and
# band (cutoff) can be scaled/ablated:
#   007a: lfm_positions [2,4,6,8] (4 blocks, uniform), mod_size 16
#   007b: lfm_positions [2,5]     (2 blocks, same as 004a), mod_size 32
# With positions [2,5] and mod_size 16 the skeleton is functionally identical
# to 004a (DEANetV3-lfm).

import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v3_arch import DEANetV3, LowFreqModBlock


@ARCH_REGISTRY.register()
class DEANetV5(DEANetV3):
    """DEA-Net with a parameterized LFM stack at the level-3 bottleneck.

    Args:
        base_dim (int): Channel number of the first level. Default: 32.
        in_ch (int): Number of input channels. Default: 3.
        out_ch (int): Number of output channels. Default: 3.
        lfm_positions (tuple): Insert an LFM after these level-3 blocks
            (1-indexed, within the 8 DEABlock chain). Default: (2, 4, 6, 8).
        mod_size (int): Spatial size of each LFM's modulation grid.
        cutoff (float): Low-pass cutoff as fraction of Nyquist.
        edge (float): Soft transition width of the mask.
    """

    def __init__(self, base_dim=32, in_ch=3, out_ch=3, lfm_positions=(2, 4, 6, 8),
                 mod_size=16, cutoff=0.08, edge=0.04):
        super(DEANetV5, self).__init__(base_dim, in_ch, out_ch, global_block='lfm')
        # drop DEANetV3's two fixed LFM slots; V5 manages its own stack
        del self.level3_fgb1, self.level3_fgb2
        dim = base_dim * 4
        self.lfm_positions = tuple(lfm_positions)
        for p in self.lfm_positions:
            if not 1 <= p <= 8:
                raise ValueError(f'lfm position {p} out of range [1, 8]')
        self.level3_lfms = nn.ModuleList(
            LowFreqModBlock(dim, cutoff=cutoff, edge=edge, mod_size=mod_size)
            for _ in self.lfm_positions)

    def forward(self, x):
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
        lfm_iter = iter(self.level3_lfms)
        for i in range(1, 9):
            h = getattr(self, f'level3_block{i}')(h)
            if i in self.lfm_positions:
                h = next(lfm_iter)(h)
        x_level3_mix = self.mix1(x_down3, h)

        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out
