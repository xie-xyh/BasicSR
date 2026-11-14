# SDAModel: 006a probe — SDA adapter training on the frozen 004a backbone.
#
# Supervised (paired) counterpart of FreqDAModel: the backbone is frozen inside
# the DEANetV4 arch, so only the SDA encoder + zero-init adapters are optimized.
# On top of DehazeModel's L1/FFT/gradient losses, adds the SDA supervision from
# Hihaze (arXiv:2610.25832, App. A.2.1): multi-scale heads predict
#   T_d: structural attenuation   1 - (Phi(I)+eps)/(Phi(I_c)+eps)      [0, 1]
#   T_r: residual correction      (Phi(I_c)-Phi(I_b))/(Phi(I_c)+eps)  [-1, 1]
#   T_s: structural confidence    sigmoid((Phi(I_c)-tau)/kappa)        (0, 1)
# where Phi combines local Sobel-gradient magnitude and local contrast,
# I is the hazy input, I_c the clean GT, and I_b the frozen backbone output.
# Config slot: train.sda_opt {loss_weight, eps_str, tau, kappa} (all optional).

from collections import OrderedDict

import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


def structure_stat(x, kernel=11):
    """Phi(x): local Sobel magnitude + local contrast, per-pixel scalar map."""
    gray = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device, dtype=x.dtype).view(1, 1, 3, 3) / 4.0
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, kx.transpose(-1, -2), padding=1)
    sob = torch.sqrt(gx**2 + gy**2 + 1e-12)
    pad = kernel // 2
    mu = F.avg_pool2d(gray, kernel, 1, pad)
    var = F.avg_pool2d(gray**2, kernel, 1, pad) - mu**2
    contrast = torch.sqrt(var.clamp(min=0))
    return F.avg_pool2d(sob, kernel, 1, pad) + contrast


@MODEL_REGISTRY.register()
class SDAModel(DehazeModel):
    """004a frozen backbone + SDA adapter training (paired data, GT available)."""

    def init_training_settings(self):
        super().init_training_settings()
        sda_opt = self.opt['train'].get('sda_opt', {})
        self.sda_weight = sda_opt.get('loss_weight', 0.0)
        self.eps_str = sda_opt.get('eps_str', 1e-3)
        self.tau = sda_opt.get('tau', 0.03)
        self.kappa = sda_opt.get('kappa', 0.01)

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

        if self.sda_weight > 0:
            with torch.no_grad():
                phi_i = structure_stat(self.lq)
                phi_c = structure_stat(self.gt)
                phi_b = structure_stat(self.net_g.last_base)
                t_d = (1 - (phi_i + self.eps_str) / (phi_c + self.eps_str)).clamp(0, 1)
                t_r = ((phi_c - phi_b) / (phi_c + self.eps_str)).clamp(-1, 1)
                t_s = torch.sigmoid((phi_c - self.tau) / self.kappa)
                target = torch.cat([t_d, t_r, t_s], dim=1)
            l_sda = 0
            for pred in self.net_g.last_sda_preds:
                tgt = F.adaptive_avg_pool2d(target, pred.shape[-2:])
                l_sda = l_sda + F.smooth_l1_loss(pred, tgt)
            l_total = l_total + self.sda_weight * l_sda
            loss_dict['l_sda'] = l_sda

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
