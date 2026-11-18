import random
from collections import OrderedDict

import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class LMHazeModel(DehazeModel):
    """Training and testing hooks for LMHaze's two-argument MoE network."""

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self.grad_clip_norm = train_opt.get('grad_clip_norm', 0)
        mixing_opt = train_opt.get('mixing_augs', {})
        self.mixup = mixing_opt.get('mixup', False)
        self.mixup_beta = mixing_opt.get('mixup_beta', 1.2)
        self.mixup_use_identity = mixing_opt.get('use_identity', False)
        self.mixup_distribution = torch.distributions.Beta(self.mixup_beta, self.mixup_beta)

    def feed_data(self, data):
        super().feed_data(data)
        logit = data.get('logit')
        self.logit = logit.to(self.device) if logit is not None else None
        if self.is_train and self.mixup and (not self.mixup_use_identity or random.random() < 0.5):
            lam = self.mixup_distribution.sample().item()
            indices = torch.randperm(self.gt.shape[0], device=self.device)
            self.gt = lam * self.gt + (1 - lam) * self.gt[indices]
            self.lq = lam * self.lq + (1 - lam) * self.lq[indices]

    def _forward(self, net, lq):
        if self.logit is None:
            return net(lq)
        return net(lq, self.logit)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self._forward(self.net_g, self.lq)

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

        l_total.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip_norm)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        height, width = self.lq.shape[2:]
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        lq = self.lq
        if pad_h or pad_w:
            lq = F.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self._forward(self.net_g_ema, lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self._forward(self.net_g, lq)
            self.net_g.train()

        if pad_h or pad_w:
            self.output = self.output[:, :, :height, :width]
