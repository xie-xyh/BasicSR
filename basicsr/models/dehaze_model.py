from collections import OrderedDict

from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY

from .sr_model import SRModel


@MODEL_REGISTRY.register()
class DehazeModel(SRModel):
    """Dehazing model: pixel loss (L1) + optional contrastive loss.

    Compared with SRModel, an extra contrastive loss term is supported via the
    `contrastive_opt` config section. The contrastive loss takes three
    arguments: (output, gt, lq), where the hazy input `lq` is used as the
    negative sample (see ContrastiveLoss in basicsr/losses/contrastive_loss.py).

    If `contrastive_opt` is not provided, DehazeModel behaves exactly like
    SRModel with `scale: 1`.
    """

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']

        # define contrastive loss
        if train_opt.get('contrastive_opt'):
            self.cri_contrastive = build_loss(train_opt['contrastive_opt']).to(self.device)
        else:
            self.cri_contrastive = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # contrastive loss (anchor=output, positive=gt, negative=hazy input)
        if self.cri_contrastive:
            l_cr = self.cri_contrastive(self.output, self.gt, self.lq)
            l_total += l_cr
            loss_dict['l_cr'] = l_cr

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
