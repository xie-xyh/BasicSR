import torch
from torch import nn

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ContrastiveLoss(nn.Module):
    """Contrastive loss in VGG19 feature space (used by DEA-Net).

    It pulls the network output (anchor) towards the ground-truth (positive)
    and pushes it away from the hazy input (negative):

        L = sum_i w_i * |f_i(a) - f_i(p)| / (|f_i(a) - f_i(n)| + eps)

    where f_i are the relu1_1 .. relu5_1 features of a frozen VGG19. Gradients
    only flow through the anchor; positive and negative features are detached.
    Note that the input images are fed to VGG in [0, 1] WITHOUT ImageNet
    normalization, which follows the original DEA-Net implementation.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        layer_weights (tuple[float]): Weights for relu1_1 to relu5_1 features.
            Default: (1/32, 1/16, 1/8, 1/4, 1).
        eps (float): A small constant for numerical stability. Default: 1e-7.
    """

    def __init__(self, loss_weight=1.0, layer_weights=(1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0), eps=1e-7):
        super(ContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps

        self.layer_name_list = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        assert len(layer_weights) == len(self.layer_name_list), (
            f'layer_weights should have {len(self.layer_name_list)} elements')
        self.layer_weights = layer_weights

        # use_input_norm=False keeps the original DEA-Net behavior (raw [0,1]
        # images instead of ImageNet-normalized inputs)
        self.vgg = VGGFeatureExtractor(
            layer_name_list=self.layer_name_list, vgg_type='vgg19', use_input_norm=False)
        self.criterion = nn.L1Loss()

    def forward(self, a, p, n):
        """Forward function.

        Args:
            a (Tensor): Anchor tensor (network output), shape (b, c, h, w).
            p (Tensor): Positive tensor (ground-truth), shape (b, c, h, w).
            n (Tensor): Negative tensor (hazy input), shape (b, c, h, w).

        Returns:
            Tensor: Contrastive loss.
        """
        a_vgg = self.vgg(a)
        p_vgg = self.vgg(p)
        n_vgg = self.vgg(n)

        loss = torch.zeros((), device=a.device)
        for i, name in enumerate(self.layer_name_list):
            d_ap = self.criterion(a_vgg[name], p_vgg[name].detach())
            d_an = self.criterion(a_vgg[name], n_vgg[name].detach())
            contrastive = d_ap / (d_an + self.eps)
            loss += self.layer_weights[i] * contrastive

        return loss * self.loss_weight
