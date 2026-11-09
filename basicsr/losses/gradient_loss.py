import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class GradientLoss(nn.Module):
    """L1 loss on Sobel gradient maps for edge supervision.

    Args:
        loss_weight (float): Default: 1.0.
        reduction (str): 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _grad(self, x):
        c = x.shape[1]
        kernel_x = self.kernel_x.expand(c, 1, 3, 3)
        kernel_y = self.kernel_y.expand(c, 1, 3, 3)
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)
        return grad_x, grad_y

    def forward(self, pred, target, weight=None, **kwargs):
        pred_gx, pred_gy = self._grad(pred)
        target_gx, target_gy = self._grad(target)
        loss = F.l1_loss(pred_gx, target_gx, reduction=self.reduction) + \
            F.l1_loss(pred_gy, target_gy, reduction=self.reduction)
        return self.loss_weight * loss
