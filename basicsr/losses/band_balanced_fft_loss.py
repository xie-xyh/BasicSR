import torch
from torch import nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class BandBalancedFFTLoss(nn.Module):
    """FFT L1 with an extra, separately normalized term for the lowest-frequency band.

    Standard FFT L1 spreads gradients almost uniformly over frequency bins, so
    the few lowest bins (global tone/veil) receive little signal (see
    research/011_wavelet_unet/NEXT_RESEARCH.md loss-gradient probe). This loss
    keeps the full-spectrum term and adds a mean over bins within a normalized
    radius, scaled so the added gradient magnitude is a fixed fraction of the
    pixel L1 gradient (011q).

    Args:
        loss_weight (float): Default: 1.0.
        low_weight (float): Relative weight of the lowest-band mean term.
            Default: 0.05.
        cutoff (float): Normalized radial frequency boundary of the lowest
            band (Nyquist = 1). Default: 0.02.
    """

    def __init__(self, loss_weight=1.0, low_weight=0.05, cutoff=0.02):
        super(BandBalancedFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.low_weight = low_weight
        self.cutoff = cutoff

    def forward(self, pred, target):
        height, width = pred.shape[-2:]
        discrepancy = torch.fft.rfft2(pred.float(), norm='ortho') - torch.fft.rfft2(
            target.float(), norm='ortho')
        magnitude = torch.view_as_real(discrepancy).abs()
        vertical = 2 * torch.fft.fftfreq(height, device=pred.device)
        horizontal = 2 * torch.fft.rfftfreq(width, device=pred.device)
        low_band = vertical[:, None].square() + horizontal[None, :].square() < self.cutoff ** 2
        full = magnitude.mean()
        low = (magnitude * low_band[..., None]).sum() / (low_band.sum() * pred.size(0) * pred.size(1) * 2)
        return self.loss_weight * (full + self.low_weight * low)
