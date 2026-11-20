import torch
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY

from .spectral_loss import FFTLoss


@LOSS_REGISTRY.register()
class SmoothResidualFFTLoss(FFTLoss):
    """FFT L1 plus an L1 on the Gaussian-smoothed residual (011s probe).

    Adds a direct penalty on the slowly varying part of the prediction error
    (residual haze veil / tone bias) with a soft frequency transition, as an
    alternative to the hard band mask of BandBalancedFFTLoss (011q). Replicate
    padding avoids periodic-wrap artifacts of FFT-based smoothing.

    Args:
        loss_weight (float): Default: 1.0.
        smooth_weight (float): Weight of the smoothed-residual term relative to
            the base FFT term (calibrated so the added gradient magnitude is a
            fixed fraction of the pixel L1 gradient). Default: 0.25.
        sigma (float): Gaussian kernel sigma in pixels. Default: 8.0.
    """

    def __init__(self, loss_weight=1.0, smooth_weight=0.25, sigma=8.0):
        super(SmoothResidualFFTLoss, self).__init__(loss_weight)
        self.smooth_weight = smooth_weight
        self.radius = round(3 * sigma)
        axis = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (axis / sigma).square())
        self.register_buffer('kernel', kernel / kernel.sum())

    def smooth(self, residual):
        channels = residual.shape[1]
        kernel = self.kernel.to(residual)
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        value = F.conv2d(F.pad(residual, (self.radius, self.radius, 0, 0), mode='replicate'),
                         horizontal, groups=channels)
        return F.conv2d(F.pad(value, (0, 0, self.radius, self.radius), mode='replicate'),
                        vertical, groups=channels)

    def forward(self, pred, target):
        base = super().forward(pred, target)
        if not self.smooth_weight:
            return base
        smooth = self.smooth(pred.float() - target.float())
        return base + self.loss_weight * self.smooth_weight * smooth.abs().mean()
