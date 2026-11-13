# FreqDA: Frequency-decoupled Dehazing Adapter (see research/005_sim2real/README.md §7).
#
# A frozen dehazing backbone (DEANetV3-LFM, 004a) plus two low-rank residual
# adapters with band-specific inductive biases:
#   y = F0(x) + R_L(x) + R_H(x_H)
#   - R_L (low band): full input -> 1/mod_size pooled -> bottleneck convs ->
#     upsample. Output is smoothly spatially varying by construction (tone /
#     airlight recalibration); zero-init.
#   - R_H (high band): bottleneck 3x3 convs on the high-pass residual x_H;
#     zero-init + TV regularization on its output (applied in FreqDAModel).
#
# The band split uses the same soft FFT mask convention as LowFreqModBlock.

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v3_arch import DEANetV3


def _soft_lowpass_mask(h, w, cutoff, edge, device):
    nyq = min(h, w) / 2.0
    v = torch.fft.fftfreq(h, device=device) * h
    u = torch.fft.rfftfreq(w, device=device) * w
    r = torch.sqrt(v[:, None]**2 + u[None, :]**2) / nyq
    return torch.sigmoid((cutoff - r) / edge)[None, None, :, :]


@ARCH_REGISTRY.register()
class FreqDA(nn.Module):
    """Frozen backbone + frequency-decoupled low-rank adapters.

    Args:
        backbone_ckpt (str): Path to the 004a checkpoint (params_ema).
        cutoff (float): Low-pass cutoff as fraction of Nyquist. Default: 0.08.
        edge (float): Soft mask transition width. Default: 0.04.
        rank_l (int): Bottleneck channels of the low-frequency adapter.
        rank_h (int): Bottleneck channels of the high-frequency adapter.
        mod_size (int): Spatial size of the low adapter's pooled input.
    """

    def __init__(self, backbone_ckpt, cutoff=0.08, edge=0.04, rank_l=16, rank_h=4, mod_size=16):
        super(FreqDA, self).__init__()
        self.cutoff = cutoff
        self.edge = edge
        self.mod_size = mod_size

        self.backbone = DEANetV3(base_dim=32, global_block='lfm')
        sd = torch.load(backbone_ckpt, map_location='cpu')
        key = 'params_ema' if 'params_ema' in sd else 'params'
        self.backbone.load_state_dict(sd[key], strict=True)
        self.backbone.requires_grad_(False)

        self.low_adapter = nn.Sequential(
            nn.Conv2d(3, rank_l, 1), nn.GELU(), nn.Conv2d(rank_l, 3, 1))
        self.high_adapter = nn.Sequential(
            nn.Conv2d(3, rank_h, 3, padding=1), nn.GELU(), nn.Conv2d(rank_h, 3, 3, padding=1))
        # zero-init both heads: initial output equals the frozen backbone exactly
        nn.init.zeros_(self.low_adapter[-1].weight)
        nn.init.zeros_(self.low_adapter[-1].bias)
        nn.init.zeros_(self.high_adapter[-1].weight)
        nn.init.zeros_(self.high_adapter[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        z = self.backbone(x)

        mask = _soft_lowpass_mask(H, W, self.cutoff, self.edge, x.device)
        spec = torch.fft.rfft2(x.float(), norm='ortho')
        x_h = torch.fft.irfft2(spec * (1 - mask), s=(H, W), norm='ortho').to(x.dtype)

        g = F.adaptive_avg_pool2d(x, self.mod_size)
        g = self.low_adapter(g)
        r_l = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)

        r_h = self.high_adapter(x_h)
        self.last_rh = r_h  # consumed by FreqDAModel for the TV loss
        return z + r_l + r_h
