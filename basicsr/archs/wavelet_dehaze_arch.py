# WaveletDehazeNet (010a): physics-decomposed prediction targets on Haar subbands.
#
# Design (research/010_wavelet/README.md): instead of regressing RGB directly,
# the network predicts the three terms of the atmospheric scattering model
# separately on a two-level Haar decomposition:
#   Module 1 (BaseBranch): LL2 base map (1/4 res) + coarse-texture energy ->
#     restored base map + spatial feature F (global context via GAP broadcast,
#     because the haze veil is a global field and the local RF of a small conv
#     stack is not enough to correct it).
#   Module 2 (TextureBranch): detail coefficients are recovered by a *bounded,
#     spatially smoothed multiplicative gain* (D_hazy ~= t * D_clear when t, A
#     are locally constant => gain ~= 1/t), not by regenerating texture.
#   Module 3: where multiplication fails (depth edges, where the veil does not
#     cancel), a DEConv-predicted small correction is added with a learned
#     weight:  D_rec = D * gain + w * corr_scale * DEConv(D).
# Output is reconstructed by exact Haar IWT back to RGB.
#
# Guards inherited from project lessons: bounded gain (exp of clamped log-gain)
# and smoothed gain maps to avoid amplifying noise/JPEG artifacts and Haar
# block artifacts; corr_scale keeps Module 3 a small correction instead of a
# second general-purpose backbone.

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_arch import DEConv
from .deanet_v6_arch import HaarDWT, HaarIWT


class BaseBranch(nn.Module):
    """LL2 base-map restoration + spatial feature for texture branches."""

    def __init__(self, feat=32):
        super(BaseBranch, self).__init__()
        # input: LL2 (3ch) + per-direction coarse texture energy (3ch)
        self.inp = nn.Sequential(
            nn.Conv2d(6, feat, 3, padding=1), nn.GELU(),
            nn.Conv2d(feat, feat, 3, padding=1), nn.GELU())
        self.gap_proj = nn.Sequential(
            nn.Conv2d(feat, feat, 1), nn.GELU(), nn.Conv2d(feat, feat, 1))
        self.base_head = nn.Conv2d(feat, 3, 3, padding=1)

    def forward(self, ll2, energy2):
        f = self.inp(torch.cat([ll2, energy2], dim=1))
        g = f.mean(dim=(2, 3), keepdim=True)
        f = f + self.gap_proj(g)
        base_rec = ll2 + self.base_head(f)
        return base_rec, f


class TextureBranch(nn.Module):
    """D_rec = D * gain + w * corr_scale * DEConv(D).

    gain: exp(avgpool3x3(clamp(log_gain, -1, 2))) -> bounded [0.37, 7.39],
    spatially smoothed. corr_scale limits the DEConv correction magnitude.
    """

    def __init__(self, feat=32, dim=9, corr_scale=0.1):
        super(TextureBranch, self).__init__()
        self.gain_net = nn.Sequential(
            nn.Conv2d(dim + feat + dim, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, dim, 3, padding=1))
        self.corr = DEConv(dim)
        self.weight = nn.Sequential(
            nn.Conv2d(feat + dim, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, dim, 1), nn.Sigmoid())
        self.corr_scale = corr_scale

    def forward(self, d, f):
        strength = F.avg_pool2d(d.abs(), 3, 1, 1)
        log_gain = self.gain_net(torch.cat([d, f, strength], dim=1))
        log_gain = F.avg_pool2d(log_gain, 3, 1, 1)
        gain = torch.exp(log_gain.clamp(-1.0, 2.0))
        corr = self.corr(d) * self.corr_scale
        w = self.weight(torch.cat([f, strength], dim=1))
        self.last_gain = gain.detach()  # 供机理分析工具读取（倍率图 ≈ 1/t）
        return d * gain + w * corr


@ARCH_REGISTRY.register()
class WaveletDehazeNet(nn.Module):
    """Two-level Haar dehazing: base-map restoration + multiplicative texture
    recovery + weighted DEConv correction.

    Args:
        feat (int): Width of the spatial feature F shared by branches.
        corr_scale (float): Fixed scale of the DEConv correction (small).
    """

    def __init__(self, feat=32, corr_scale=0.1):
        super(WaveletDehazeNet, self).__init__()
        self.dwt1 = HaarDWT(3)
        self.dwt2 = HaarDWT(3)
        self.iwt2 = HaarIWT(3)
        self.iwt1 = HaarIWT(3)
        self.base_branch = BaseBranch(feat)
        self.texture_coarse = TextureBranch(feat, corr_scale=corr_scale)
        self.texture_fine = TextureBranch(feat, corr_scale=corr_scale)

    @staticmethod
    def _split_subbands(x, ch=3):
        # HaarDWT 输出布局：逐输入通道交错 (LL, LH, HL, HH)
        B, _, h, w = x.shape
        x = x.view(B, ch, 4, h, w)
        ll = x[:, :, 0]
        det = torch.cat([x[:, :, 1], x[:, :, 2], x[:, :, 3]], dim=1)  # (B, 3ch, h, w)
        return ll, det

    @staticmethod
    def _pack_subbands(ll, det, ch=3):
        B, _, h, w = ll.shape
        lh, hl, hh = det[:, 0:ch], det[:, ch:2 * ch], det[:, 2 * ch:3 * ch]
        return torch.stack([ll, lh, hl, hh], dim=2).view(B, 4 * ch, h, w)

    def forward(self, x):
        B, _, H, W = x.shape
        ll1, det1 = self._split_subbands(self.dwt1(x))
        ll2, det2 = self._split_subbands(self.dwt2(ll1))

        # 每方向纹理能量（对 RGB 求均值），供底图分支判断"亮而平坦≠浓雾"
        energy2 = F.avg_pool2d(det2.abs(), 3, 1, 1).view(B, 3, 3, *ll2.shape[-2:]).mean(dim=1)
        base_rec, f = self.base_branch(ll2, energy2)

        det2_rec = self.texture_coarse(det2, f)
        f1 = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
        det1_rec = self.texture_fine(det1, f1)

        ll1_rec = self.iwt2(self._pack_subbands(base_rec, det2_rec))
        out = self.iwt1(self._pack_subbands(ll1_rec, det1_rec), out_size=(H, W))
        return out
