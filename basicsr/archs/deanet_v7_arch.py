# DEANetV7: DEA-Net + low-frequency processing branch with gated fusion (009a).
#
# Motivation (research/009_freqbranch/README.md): 004a validated that a global
# low-frequency pathway helps (LFM, +0.13 dB), but the pathway is a thin FiLM
# modulator. V6 replaces each LowFreqModBlock with an actual low-frequency
# *branch*: split the low band with the same soft FFT mask, process it at a
# coarse spatial grid (32x32, where a 3x3 conv stack has a global receptive
# field by construction), then fuse back through a per-pixel content gate.
#
# Design constraints inherited from 003d/004b/006a/007a failures:
#   - main path keeps full-band features (no hard band routing of the trunk);
#   - branch output projection is zero-initialized, so the whole network starts
#     exactly as DEA-Net without the branch (identity at init);
#   - gate is a plain learned conv (no exotic parameterization).
#
# The two branch blocks sit at the same level-3 positions as 004a's LFMs
# (after block2 and block5).

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v3_arch import DEANetV3, LowFreqModBlock


class LowFreqBranchBlock(nn.Module):
    """Low-frequency processing branch with per-pixel gated fusion.

    x -> low band (soft FFT mask) -> pool to branch_res -> conv stack ->
    upsample -> y_L;  out = x + sigmoid(conv([x, y_L])) * y_L

    Args:
        dim (int): Channel number.
        cutoff (float): Low-pass cutoff as fraction of Nyquist.
        edge (float): Soft mask transition width.
        branch_res (int): Spatial size of the coarse processing grid.
        n_convs (int): Number of 3x3 convs in the coarse branch stack.
    """

    def __init__(self, dim, cutoff=0.08, edge=0.04, branch_res=32, n_convs=3):
        super(LowFreqBranchBlock, self).__init__()
        self.cutoff = cutoff
        self.edge = edge
        self.branch_res = branch_res
        convs = []
        for _ in range(n_convs - 1):
            convs += [nn.Conv2d(dim, dim, 3, padding=1), nn.GELU()]
        convs += [nn.Conv2d(dim, dim, 3, padding=1)]
        self.branch = nn.Sequential(*convs)
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)
        self.gate = nn.Conv2d(dim * 2, dim, 3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        mask = LowFreqModBlock._soft_lowpass_mask(
            H, W, self.cutoff, self.edge, x.device, torch.float32)
        spec = torch.fft.rfft2(x.float(), norm='ortho')
        low = torch.fft.irfft2(spec * mask, s=(H, W), norm='ortho').to(x.dtype)

        y = F.adaptive_avg_pool2d(low, self.branch_res)
        y = self.branch(y)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
        g = torch.sigmoid(self.gate(torch.cat([x, y], dim=1)))
        return x + g * y


@ARCH_REGISTRY.register()
class DEANetV7(DEANetV3):
    """DEA-Net with low-frequency branch blocks at the level-3 bottleneck.

    Same skeleton as 004a except the two LFM slots (after level-3 block2 and
    block5) are replaced by LowFreqBranchBlock.

    Args:
        base_dim (int): Channel number of the first level. Default: 32.
        in_ch (int): Number of input channels. Default: 3.
        out_ch (int): Number of output channels. Default: 3.
        branch_res (int): Coarse processing grid size of the low-freq branch.
        n_convs (int): Number of convs in the branch stack.
    """

    def __init__(self, base_dim=32, in_ch=3, out_ch=3, branch_res=32, n_convs=3):
        super(DEANetV7, self).__init__(base_dim, in_ch, out_ch, global_block='lfm')
        del self.level3_fgb1, self.level3_fgb2
        dim = base_dim * 4
        self.level3_lfb1 = LowFreqBranchBlock(dim, branch_res=branch_res, n_convs=n_convs)
        self.level3_lfb2 = LowFreqBranchBlock(dim, branch_res=branch_res, n_convs=n_convs)

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
        h = self.level3_block1(h)
        h = self.level3_block2(h)
        h = self.level3_lfb1(h)
        h = self.level3_block3(h)
        h = self.level3_block4(h)
        h = self.level3_block5(h)
        h = self.level3_lfb2(h)
        h = self.level3_block6(h)
        h = self.level3_block7(h)
        h = self.level3_block8(h)
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
