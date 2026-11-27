"""MaritimeSFModel (014): DehazeModel + optional multi-scale deep supervision and diagnostics."""
from collections import OrderedDict

import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class MaritimeSFModel(DehazeModel):
    """DehazeModel subclass for MaritimeSFNet (014a / 014b / 014c).

    Adds two non-intrusive capabilities on top of DehazeModel:
        1. Optional training-only multi-scale deep supervision (`train.deep_sup_opt`)
           on the 1/4 and 1/2 auxiliary BASR predictions exposed in
           `net_g.aux_outputs` (014c; disabled in 014a/b when absent).
        2. Per-iteration logging of scalar internal diagnostics (`basr_gain`,
           `basr_bias`, `fcsf3_alpha`, `fcsf2_alpha`, `fcsf1_beta`) to catch
           gate collapse or gain saturation early.
    """

    def init_training_settings(self):
        super().init_training_settings()
        ds_opt = self.opt['train'].get('deep_sup_opt') or {}
        default_w = float(ds_opt.get('loss_weight', 0.0))
        raw_weights = ds_opt.get('weights')
        if raw_weights is not None:
            self.deep_sup_weights = {str(k): float(v) for k, v in raw_weights.items()}
        elif default_w > 0:
            self.deep_sup_weights = {'1/4': default_w, '1/2': default_w}
        else:
            self.deep_sup_weights = {}
        self.deep_sup_fft_weight = float(ds_opt.get('fft_weight', 1.0))

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

        bare = self.get_bare_model(self.net_g)
        aux_outputs = getattr(bare, 'aux_outputs', None) or {}
        if self.deep_sup_weights and aux_outputs:
            h, w = self.gt.shape[-2:]
            pad_h = (4 - h % 4) % 4
            pad_w = (4 - w % 4) % 4
            gt_pad = (
                F.pad(self.gt, (0, pad_w, 0, pad_h), mode='replicate')
                if (pad_h or pad_w) else self.gt)
            for scale_key, pred_aux in aux_outputs.items():
                w_scale = self.deep_sup_weights.get(scale_key, 0.0)
                if w_scale <= 0:
                    continue
                gt_s = F.interpolate(
                    gt_pad, size=pred_aux.shape[-2:], mode='bilinear', align_corners=False)
                l_aux = F.l1_loss(pred_aux, gt_s)
                if self.cri_fft and self.deep_sup_fft_weight > 0:
                    l_aux = l_aux + self.deep_sup_fft_weight * self.cri_fft(pred_aux, gt_s)
                l_total = l_total + w_scale * l_aux
                loss_dict[f"l_aux_{scale_key.replace('/', '_')}"] = l_aux

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        diagnostics = getattr(bare, 'diagnostics', None) or {}
        for key, value in diagnostics.items():
            if value is not None:
                self.log_dict[key] = value.item() if torch.is_tensor(value) else float(value)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
