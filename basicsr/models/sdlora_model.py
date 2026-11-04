import os
import time

import torch
from torch.nn import functional as F

from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only
from basicsr.utils.registry import MODEL_REGISTRY

from .dehaze_model import DehazeModel


@MODEL_REGISTRY.register()
class SDLoRAModel(DehazeModel):
    """SR-style training for SDLoRADehaze (frozen SD2.1-base + UNet LoRA).

    Differences from DehazeModel:
    - test() reflect-pads to multiples of 8 (VAE scale factor) instead of 4;
    - save() stores only trainable parameters (LoRA + optionally unfrozen
      blocks), so checkpoints stay small even with the frozen SD base;
    - load_network() only applies keys matching the current trainable set
      (the frozen SD base always comes from the local sd_path snapshot), so
      resume/test need `strict_load_g: false`;
    - model_ema() only tracks trainable parameters;
    - print_network()/setup_optimizers() log summaries instead of dumping the
      whole SD structure / hundreds of frozen-parameter warnings.
    """

    @staticmethod
    def _trainable_keys(net):
        return {k for k, v in net.named_parameters() if v.requires_grad}

    def _trainable_state_dict(self, net):
        keys = self._trainable_keys(net)
        return {k: v.cpu() for k, v in net.state_dict().items() if k in keys}

    def test(self):
        h, w = self.lq.shape[2:]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
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

    def model_ema(self, decay=0.999):
        trainable = self._trainable_keys(self.get_bare_model(self.net_g))
        net_g_params = dict(self.get_bare_model(self.net_g).named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())
        for k in trainable:
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = [v for _, v in self.net_g.named_parameters() if v.requires_grad]

        logger = get_root_logger()
        total = sum(p.numel() for p in self.net_g.parameters())
        trainable = sum(p.numel() for p in optim_params)
        logger.info(f'SDLoRAModel: trainable LoRA params {trainable:,d} / total {total:,d} '
                    f'({100 * trainable / total:.2f}%).')

        optim_opt = dict(train_opt['optim_g'])
        optim_type = optim_opt.pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **optim_opt)
        self.optimizers.append(self.optimizer_g)

    @master_only
    def print_network(self, net):
        bare = self.get_bare_model(net)
        n_total = sum(p.numel() for p in bare.parameters())
        n_train = sum(p.numel() for p in bare.parameters() if p.requires_grad)
        logger = get_root_logger()
        logger.info(f'Network: {bare.__class__.__name__} (SD2.1-base frozen + UNet LoRA), '
                    f'total params: {n_total:,d}, trainable: {n_train:,d}')

    def load_network(self, net, load_path, strict=True, param_key='params'):
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        # keep only keys belonging to the current trainable set (LoRA and/or
        # unfrozen blocks); missing keys (e.g. newly added conv LoRA when
        # warm-starting from an attention-only checkpoint) stay at init values
        trainable = self._trainable_keys(net)
        trainable_net = {k: v for k, v in load_net.items()
                         if (k[7:] if k.startswith('module.') else k) in trainable}
        trainable_net = {(k[7:] if k.startswith('module.') else k): v for k, v in trainable_net.items()}
        net.load_state_dict(trainable_net, strict=False)
        logger.info(f'Loading {net.__class__.__name__} trainable params from {load_path} '
                    f'[param key: {param_key}]: {len(trainable_net)} tensors loaded.')

    @master_only
    def _save_trainable_network(self, current_iter):
        if current_iter == -1:
            current_iter = 'latest'
        save_path = os.path.join(self.opt['path']['models'], f'net_g_{current_iter}.pth')
        save_dict = {'params': self._trainable_state_dict(self.get_bare_model(self.net_g))}
        if hasattr(self, 'net_g_ema'):
            save_dict['params_ema'] = self._trainable_state_dict(self.get_bare_model(self.net_g_ema))

        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger = get_root_logger()
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def save(self, epoch, current_iter):
        self._save_trainable_network(current_iter)
        self.save_training_state(epoch, current_iter)
