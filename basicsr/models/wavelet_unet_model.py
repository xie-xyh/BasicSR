# WaveletUNetModel (011a): DehazeModel + image-space subband supervision.
#
# aux from WaveletUNet: (ll1_rec, det1_rec) — restored LL1 base map and
# detail1 coefficients. GT targets are computed with the network's own HaarDWT
# on the GT (level 1 only). Config slot: train.subband_opt
# {loss_weight, w_base, w_detail}.

from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class WaveletUNetModel(DehazeModel):
    """DehazeModel + direct L1 on LL1 / detail1 subband predictions."""

    def init_training_settings(self):
        super().init_training_settings()
        sub_opt = self.opt['train'].get('subband_opt', {})
        self.subband_weight = sub_opt.get('loss_weight', 0.0)
        self.w_base = sub_opt.get('w_base', 1.0)
        self.w_detail = sub_opt.get('w_detail', 1.0)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.cri_contrastive:
            l_cr = self.cri_contrastive(self.output, self.gt, self.lq)
            l_total += l_cr
            loss_dict['l_cr'] = l_cr
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
        if self.cri_grad:
            l_grad = self.cri_grad(self.output, self.gt)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        if self.subband_weight > 0:
            ll1_rec, det1_rec = self.net_g.aux
            ll1_gt, det1_gt = self.net_g._split_subbands(self.net_g.dwt1(self.gt))
            l_base = (ll1_rec - ll1_gt).abs().mean()
            l_detail = (det1_rec - det1_gt).abs().mean()
            l_sub = self.w_base * l_base + self.w_detail * l_detail
            l_total = l_total + self.subband_weight * l_sub
            loss_dict['l_base'] = l_base
            loss_dict['l_detail'] = l_detail

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
