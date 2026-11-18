# WaveletDehazeModel (010b): DehazeModel + subband-decomposed supervision.
#
# Motivation: with only the end-to-end RGB loss, the texture branches' gradients
# are ~400x smaller than the base branch's (measured), so each module learns
# slowly and its errors are entangled. Add direct L1 supervision on the three
# prediction targets (LL2 base map, coarse detail2, fine detail1), computed
# against the Haar decomposition of the GT using the network's own DWT buffers.
#
# Config slot: train.subband_opt {loss_weight, w_base, w_coarse, w_fine}.
# Weights are relative to a shared loss_weight; defaults 1.0 each.

from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class WaveletDehazeModel(DehazeModel):
    """DehazeModel + direct L1 supervision on the Haar subband targets."""

    def init_training_settings(self):
        super().init_training_settings()
        sub_opt = self.opt['train'].get('subband_opt', {})
        self.subband_weight = sub_opt.get('loss_weight', 0.0)
        self.w_base = sub_opt.get('w_base', 1.0)
        self.w_coarse = sub_opt.get('w_coarse', 1.0)
        self.w_fine = sub_opt.get('w_fine', 1.0)

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
            base_rec, det2_rec, det1_rec = self.net_g.aux
            ll1_gt, det1_gt = self.net_g._split_subbands(self.net_g.dwt1(self.gt))
            ll2_gt, det2_gt = self.net_g._split_subbands(self.net_g.dwt2(ll1_gt))
            l_base = (base_rec - ll2_gt).abs().mean()
            l_coarse = (det2_rec - det2_gt).abs().mean()
            l_fine = (det1_rec - det1_gt).abs().mean()
            l_sub = self.w_base * l_base + self.w_coarse * l_coarse + self.w_fine * l_fine
            l_total = l_total + self.subband_weight * l_sub
            loss_dict['l_base'] = l_base
            loss_dict['l_coarse'] = l_coarse
            loss_dict['l_fine'] = l_fine

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
