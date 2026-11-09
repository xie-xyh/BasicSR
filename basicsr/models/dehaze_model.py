from collections import OrderedDict

import torch
from torch.nn import functional as F

from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY

from .sr_model import SRModel


@MODEL_REGISTRY.register()
class DehazeModel(SRModel):
    """Dehazing model: pixel loss (L1) + optional contrastive / FFT / gradient losses.

    Compared with SRModel, extra loss terms are supported via config sections:
    `contrastive_opt` takes (output, gt, lq) with the hazy input as negative
    sample; `fft_opt` (frequency-domain, e.g. FFTLoss) and `gradient_opt`
    (edge, e.g. GradientLoss) both take (output, gt).

    If `contrastive_opt` is not provided, DehazeModel behaves exactly like
    SRModel with `scale: 1`.

    Note: DEANet contains two stride-2 downsamples followed by transposed-conv
    upsamples, so its skip connections only align when H and W are multiples
    of 4. test() therefore reflect-pads arbitrary-size inputs to multiples of
    4 and crops the padding off afterwards.
    """

    def test(self):
        h, w = self.lq.shape[2:]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        lq = self.lq
        if pad_h != 0 or pad_w != 0:
            lq = F.pad(lq, (0, pad_w, 0, pad_h), mode='reflect')

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lq)
            self.net_g.train()

        if pad_h != 0 or pad_w != 0:
            self.output = self.output[:, :, :h, :w]

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']

        # define contrastive loss
        if train_opt.get('contrastive_opt'):
            self.cri_contrastive = build_loss(train_opt['contrastive_opt']).to(self.device)
        else:
            self.cri_contrastive = None

        # define frequency-domain loss
        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        # define gradient (edge) loss
        if train_opt.get('gradient_opt'):
            self.cri_grad = build_loss(train_opt['gradient_opt']).to(self.device)
        else:
            self.cri_grad = None

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
        # frequency-domain loss
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
        # gradient (edge) loss
        if self.cri_grad:
            l_grad = self.cri_grad(self.output, self.gt)
            l_total += l_grad
            loss_dict['l_grad'] = l_grad

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
