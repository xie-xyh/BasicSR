from torch import nn

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class LogMSELoss(nn.Module):
    """Log of per-image MSE, averaged over the batch (011t probe).

    The log compresses the loss range, giving relatively more weight to
    already-good images than plain MSE does.

    Args:
        loss_weight (float): Default: 1.0.
        mse_floor (float): Constant added before the log to avoid log(0) and
            to set the gradient scale. Default: 1e-4.
    """

    def __init__(self, loss_weight=1.0, mse_floor=1e-4):
        super(LogMSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.mse_floor = mse_floor

    def forward(self, pred, target):
        error = pred.float() - target.float()
        mse = error.square().flatten(1).mean(dim=1)
        return self.loss_weight * (mse + self.mse_floor).log().mean()
