import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    """L1 loss between FFT spectra (real & imaginary parts) of pred and target.

    Pixel-space L1 under-constrains global tone/illumination, which live in
    low-frequency bins; this term penalizes frequency-domain errors directly
    and complements pixel losses for haze removal.

    Args:
        loss_weight (float): Default: 1.0.
        reduction (str): 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        pred_fft = torch.fft.rfft2(pred.float(), norm='ortho')
        target_fft = torch.fft.rfft2(target.float(), norm='ortho')
        pred_spec = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_spec = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * F.l1_loss(pred_spec, target_spec, reduction=self.reduction)
