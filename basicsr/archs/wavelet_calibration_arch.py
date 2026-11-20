import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .wavelet_unet_arch import WaveletUNet


@ARCH_REGISTRY.register()
class CalibratedWaveletUNet(WaveletUNet):
    """Frozen WaveletUNet plus a trainable global color-calibration head (011l).

    The backbone is frozen (requires_grad_(False) is called before the head is
    created, so the head stays trainable). The head reads the hazy input and
    predicts a per-channel affine correction whose coefficients vary linearly
    along the image height:

        out = restored + offset + slope * linspace(-1, 1, H)

    where offset/slope each combine the restored RGB and a constant basis.

    Args:
        **kwargs: Forwarded to WaveletUNet.
    """

    def __init__(self, **kwargs):
        super(CalibratedWaveletUNet, self).__init__(**kwargs)
        self.requires_grad_(False)
        self.color_head = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 2)), nn.Flatten(),
            nn.Linear(48 * 8, 24))
        nn.init.zeros_(self.color_head[-1].weight)
        nn.init.zeros_(self.color_head[-1].bias)

    def forward(self, image):
        with torch.no_grad():
            restored = super().forward(image)
        signal = self.color_head(F.adaptive_avg_pool2d(image, (32, 32)))
        signal = 0.2 * signal.tanh().reshape(-1, 2, 3, 4)
        bias_input = torch.ones_like(restored[:, :1])
        basis = torch.cat((restored, bias_input), dim=1)
        offset = torch.einsum('boc,bchw->bohw', signal[:, 0], basis)
        slope = torch.einsum('boc,bchw->bohw', signal[:, 1], basis)
        height = torch.linspace(-1, 1, restored.shape[-2], device=image.device, dtype=image.dtype)
        return restored + offset + slope * height[None, None, :, None]
