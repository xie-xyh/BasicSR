# FreqDAModel: self-supervised adapter training on unpaired real hazy images.
#
# The backbone is frozen (FreqDA arch), so only adapter parameters are
# optimized. Losses are physical priors (no GT): dark-channel prior, channel
# dispersion floor, and TV on the high-frequency adapter's output.
# Config slots: train.dcp_opt / dispersion_opt / tv_opt (all optional).

from collections import OrderedDict

from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class FreqDAModel(DehazeModel):
    """FreqDA adapter training model (no GT; prior losses only)."""

    def init_training_settings(self):
        train_opt = self.opt['train']
        # 无 GT 适配训练没有 pixel/perceptual loss，SRModel 的
        # 'Both pixel and perceptual losses are None' 检查对它不适用：
        # 注入一个恒零的 L1 占位（永不参与计算，仅为过该检查）
        if not train_opt.get('pixel_opt') and not train_opt.get('perceptual_opt'):
            train_opt = dict(train_opt)
            train_opt['pixel_opt'] = dict(type='L1Loss', loss_weight=0.0, reduction='mean')
            self.opt['train'] = train_opt
        super().init_training_settings()

        if train_opt.get('dcp_opt'):
            self.cri_dcp = build_loss(train_opt['dcp_opt']).to(self.device)
        else:
            self.cri_dcp = None

        if train_opt.get('dispersion_opt'):
            self.cri_disp = build_loss(train_opt['dispersion_opt']).to(self.device)
        else:
            self.cri_disp = None

        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None

        if train_opt.get('color_anchor_opt'):
            self.cri_anchor = build_loss(train_opt['color_anchor_opt']).to(self.device)
        else:
            self.cri_anchor = None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # 占位 pixel loss 权重为 0，跳过计算（无 GT）
        if self.cri_dcp:
            l_dcp = self.cri_dcp(self.output)
            l_total += l_dcp
            loss_dict['l_dcp'] = l_dcp
        if self.cri_disp:
            l_disp = self.cri_disp(self.output)
            l_total += l_disp
            loss_dict['l_disp'] = l_disp
        if self.cri_tv:
            l_tv = self.cri_tv(self.net_g.last_rh)
            l_total += l_tv
            loss_dict['l_tv'] = l_tv
        if self.cri_anchor:
            l_anchor = self.cri_anchor(self.output, self.net_g.last_z.detach())
            l_total += l_anchor
            loss_dict['l_anchor'] = l_anchor

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
