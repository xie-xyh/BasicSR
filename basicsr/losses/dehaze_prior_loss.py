# Self-supervised physical-prior losses for real-haze adapter training (no GT).
#
# - DarkChannelLoss: haze brightens the dark channel; push it down (careful with
#   the weight: over-pushing produces unnaturally dark outputs).
# - DispersionFloorLoss: haze crushes per-pixel RGB channel dispersion (measured:
#   synthetic hazy 0.018 vs clean GT 0.114 on OWH); encourage a floor tau.
# - TVLoss: total variation of an adapter's output residual, discouraging
#   noise-amplifying high-frequency corrections.

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DarkChannelLoss(nn.Module):
    """Dark channel prior loss with a floor: relu(mean(dark) - floor).

    The raw (unbounded) version lets the optimizer push the dark channel
    negative, i.e. over-darken/re-color the output to cheat the metric
    (observed in the 005a probe: cyan cast, FADE cheated to 0.91). The floor
    stops the pressure once the dark channel is already low enough.

    Args:
        loss_weight (float): Loss weight.
        patch (int): Min-pooling window. Default: 7.
        floor (float): Dark-channel floor (0-1 scale). Default: 0.1.
    """

    def __init__(self, loss_weight=1.0, patch=7, floor=0.1):
        super(DarkChannelLoss, self).__init__()
        self.loss_weight = loss_weight
        self.patch = patch
        self.floor = floor

    def forward(self, x):
        dark = x.min(dim=1, keepdim=True)[0]
        dark = -F.max_pool2d(-dark, self.patch, stride=1, padding=self.patch // 2)
        return F.relu(dark.mean() - self.floor) * self.loss_weight


@LOSS_REGISTRY.register()
class ColorAnchorLoss(nn.Module):
    """Per-channel mean anchor to a reference (e.g. frozen backbone output).

    Keeps the adapter's global color drift bounded while leaving local /
    spatially-varying corrections free. FreqDAModel feeds (output, z_ref).

    Args:
        loss_weight (float): Loss weight.
    """

    def __init__(self, loss_weight=1.0):
        super(ColorAnchorLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, ref):
        return (x.mean(dim=(2, 3)) - ref.mean(dim=(2, 3))).abs().mean() * self.loss_weight


@LOSS_REGISTRY.register()
class DispersionFloorLoss(nn.Module):
    """Channel-dispersion floor: relu(tau - mean(max_c - min_c)).

    Args:
        loss_weight (float): Loss weight.
        tau (float): Dispersion floor (0-1 scale). Default: 0.08.
    """

    def __init__(self, loss_weight=1.0, tau=0.08):
        super(DispersionFloorLoss, self).__init__()
        self.loss_weight = loss_weight
        self.tau = tau

    def forward(self, x):
        disp = (x.max(dim=1)[0] - x.min(dim=1)[0]).mean()
        return F.relu(torch.as_tensor(self.tau, device=x.device) - disp) * self.loss_weight


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    """Total variation loss (spatial smoothness of a residual map).

    Args:
        loss_weight (float): Loss weight.
    """

    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        return (dx + dy) * self.loss_weight
