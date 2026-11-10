# DEANetV2: DEA-Net + frequency-domain global context at the bottleneck.
#
# Motivation (see research/003_deanet_global/README.md §2): on OWH Dense, DEA-Net's
# failures are dominated by global low-frequency errors (large-area residual haze,
# global tone shift) rather than high-frequency edges, while its DEConv blocks are
# high-pass by design and the bottleneck stack has a limited receptive field.
#
# The added FreqGlobalBlock follows GFNet (Rao et al., NeurIPS 2021) /
# SFHformer (Jiang et al., ECCV 2024): a learnable complex filter applied to the
# rFFT2 spectrum gives true global receptive field at negligible cost, which is
# exactly where the smooth haze/transmission component lives. The filter is stored
# at a canonical size and bilinearly resized to the actual spectrum shape, so the
# block works for arbitrary input resolutions (train crops vs full-size test).

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_arch import CGAFusion, DEABlock, DEBlock, default_conv


class FreqGlobalBlock(nn.Module):
    """Learnable global frequency filter block (GFNet-style) with residual FFN.

    Args:
        dim (int): Channel number.
        filter_h (int): Canonical stored height of the frequency filter.
        filter_w (int): Canonical stored width (full, not rfft-truncated) of the
            filter; the rfft layout keeps `filter_w // 2 + 1` columns.
    """

    def __init__(self, dim, filter_h=32, filter_w=32):
        super(FreqGlobalBlock, self).__init__()
        # complex filter stored as real/imag in the last dim; zero-init keeps the
        # block near-identity at start so it slots into a pretrained-style stack
        self.norm1 = nn.GroupNorm(1, dim)
        self.filter_weight = nn.Parameter(torch.zeros(dim, filter_h, filter_w // 2 + 1, 2))
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=True), nn.GELU(), nn.Conv2d(dim * 2, dim, 1, bias=True))

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        x = self.norm1(x)
        spec = torch.fft.rfft2(x.float(), norm='ortho')  # B, C, H, W//2+1 complex
        weight = torch.view_as_complex(self.filter_weight)  # C, h, w//2+1
        weight = F.interpolate(
            torch.stack([weight.real, weight.imag], dim=-1).permute(3, 0, 1, 2),  # 2, C, h, w'
            size=(H, W // 2 + 1), mode='bilinear', align_corners=False)
        weight = torch.view_as_complex(weight.permute(1, 2, 3, 0).contiguous())  # C, H, W//2+1
        spec = spec * weight.unsqueeze(0)
        x = torch.fft.irfft2(spec, s=(H, W), norm='ortho').to(res.dtype)
        x = res + x
        x = x + self.ffn(self.norm2(x))
        return x


@ARCH_REGISTRY.register()
class DEANetV2(nn.Module):
    """DEA-Net with FreqGlobalBlocks interleaved in the level-3 bottleneck.

    Structure identical to DEANet except two global blocks inserted after
    level3_block2 and level3_block5 (DEA, DEA, FGB, DEA, DEA, FGB, DEA, DEA).

    Args:
        base_dim (int): Channel number of the first level. Default: 32.
        in_ch (int): Number of input channels. Default: 3.
        out_ch (int): Number of output channels. Default: 3.
    """

    def __init__(self, base_dim=32, in_ch=3, out_ch=3):
        super(DEANetV2, self).__init__()
        # down-sample
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, base_dim, kernel_size=3, stride=1, padding=1))
        self.down2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        self.down3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1), nn.ReLU(True))
        # level1
        self.down_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.down_level1_block4 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block1 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block2 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block3 = DEBlock(default_conv, base_dim, 3)
        self.up_level1_block4 = DEBlock(default_conv, base_dim, 3)
        # level2
        self.fe_level_2 = nn.Conv2d(
            in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = DEBlock(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = DEBlock(default_conv, base_dim * 2, 3)
        # level3
        self.fe_level_3 = nn.Conv2d(
            in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block2 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_fgb1 = FreqGlobalBlock(base_dim * 4)
        self.level3_block3 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block4 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block5 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_fgb2 = FreqGlobalBlock(base_dim * 4)
        self.level3_block6 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block7 = DEABlock(default_conv, base_dim * 4, 3)
        self.level3_block8 = DEABlock(default_conv, base_dim * 4, 3)
        # up-sample
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, out_ch, kernel_size=3, stride=1, padding=1))
        # feature fusion
        self.mix1 = CGAFusion(base_dim * 4, reduction=8)
        self.mix2 = CGAFusion(base_dim * 2, reduction=4)

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
        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)
        out = self.up3(x_up2)

        return out
