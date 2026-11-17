# DEANetV6: DEANetV5 skeleton with Haar wavelet down/up-sampling (008 series).
#
# Motivation (research/008_dwt/README.md): 004a's residual-spectrum analysis
# shows the >0.50 Nyquist band is the *least* recovered band (63% recovery vs
# 96% in DC~0.02), i.e. fine detail is under-served. One suspect is the
# lossy stride-2 conv / transposed-conv pair at the two resolution transitions
# (down2/down3, up1/up2): high-frequency detail can be aliased away before the
# level-3 bottleneck and never restored. V6 replaces them with an orthogonal
# Haar DWT/IWT pair, which is lossless by construction (IWT(DWT(x)) == x), so
# detail survives the encode/decode round trip; a 1x1 conv mixes the four
# sub-bands channel-wise, keeping the parameter count aligned with the
# original 3x3 stride-conv (research/004_lowfreq single-variable protocol).
# Only the sampling operators change vs DEANetV5; with lfm_positions (2, 5)
# the rest of the network is identical to 004a.

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v5_arch import DEANetV5


def _haar_weight(ch):
    """Orthonormal Haar filters as a depthwise conv weight of shape (4C, 1, 2, 2).

    Output channel order per input channel: (LL, LH, HL, HH).
    """
    ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    lh = torch.tensor([[0.5, 0.5], [-0.5, -0.5]])
    hl = torch.tensor([[0.5, -0.5], [0.5, -0.5]])
    hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    return torch.stack([ll, lh, hl, hh]).unsqueeze(1).repeat(ch, 1, 1, 1)


class HaarDWT(nn.Module):
    """(B, C, H, W) -> (B, 4C, H/2, W/2), lossless. Pads by 1 (replicate) if odd."""

    def __init__(self, ch):
        super(HaarDWT, self).__init__()
        self.ch = ch
        self.register_buffer('weight', _haar_weight(ch))

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
        return F.conv2d(x, self.weight.to(x.dtype), stride=2, groups=self.ch)


class HaarIWT(nn.Module):
    """(B, 4C, H, W) -> (B, C, 2H, 2W); exact inverse of HaarDWT (before padding)."""

    def __init__(self, ch):
        super(HaarIWT, self).__init__()
        self.ch = ch
        self.register_buffer('weight', _haar_weight(ch))

    def forward(self, x, out_size=None):
        out = F.conv_transpose2d(x, self.weight.to(x.dtype), stride=2, groups=self.ch)
        if out_size is not None:
            out = out[:, :, :out_size[0], :out_size[1]]
        return out


class WaveletDown(nn.Module):
    """Drop-in replacement for a stride-2 conv + ReLU: DWT -> 1x1 conv -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super(WaveletDown, self).__init__()
        self.dwt = HaarDWT(in_ch)
        self.proj = nn.Conv2d(in_ch * 4, out_ch, kernel_size=1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.proj(self.dwt(x)))


class WaveletUp(nn.Module):
    """Drop-in replacement for transposed conv + ReLU: 1x1 conv -> IWT -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super(WaveletUp, self).__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1)
        self.iwt = HaarIWT(out_ch)
        self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.iwt(self.proj(x)))


@ARCH_REGISTRY.register()
class DEANetV6(DEANetV5):
    """DEANetV5 with wavelet down/up-sampling at the two resolution transitions.

    Replaces down2/down3 (stride-2 conv) with WaveletDown and up1/up2
    (transposed conv) with WaveletUp; down1/up3 and all blocks are unchanged.

    Args: same as DEANetV5. With lfm_positions (2, 5), mod_size 16, cutoff
    0.08 everything except the sampling operators matches 004a.
    """

    def __init__(self, base_dim=32, in_ch=3, out_ch=3, lfm_positions=(2, 5),
                 mod_size=16, cutoff=0.08, edge=0.04):
        super(DEANetV6, self).__init__(base_dim, in_ch, out_ch, lfm_positions,
                                       mod_size, cutoff, edge)
        self.down2 = WaveletDown(base_dim, base_dim * 2)
        self.down3 = WaveletDown(base_dim * 2, base_dim * 4)
        self.up1 = WaveletUp(base_dim * 4, base_dim * 2)
        self.up2 = WaveletUp(base_dim * 2, base_dim)
