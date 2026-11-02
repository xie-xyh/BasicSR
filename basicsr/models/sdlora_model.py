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
    - save() stores only LoRA parameters (MB-scale checkpoints instead of
      multi-GB full SD dumps);
    - load_network() only applies LoRA keys (the frozen SD base always comes
      from the local sd_path snapshot), so resume/test need
      `strict_load_g: false`;
    - model_ema() only tracks LoRA parameters;
    - print_network()/setup_optimizers() log summaries instead of dumping the
      whole SD structure / hundreds of frozen-parameter warnings.
    """

    @staticmethod
    def _lora_state_dict(net):
        state_dict = net.state_dict()
        return {k: v.cpu() for k, v in state_dict.items() if 'lora_' in k}

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
        net_g_params = dict(self.get_bare_model(self.net_g).named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())
        for k, ema_v in net_g_ema_params.items():
            if 'lora_' in k:
                ema_v.data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

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
        lora_keys = {}
        for k, v in load_net.items():
            if k.startswith('module.'):
                k = k[7:]
            if 'lora_' in k:
                lora_keys[k] = v
        net.load_state_dict(lora_keys, strict=False)
        logger.info(f'Loading {net.__class__.__name__} LoRA params from {load_path} '
                    f'[param key: {param_key}]: {len(lora_keys)} tensors loaded.')

    @master_only
    def _save_lora_network(self, current_iter):
        if current_iter == -1:
            current_iter = 'latest'
        save_path = os.path.join(self.opt['path']['models'], f'net_g_{current_iter}.pth')
        save_dict = {'params': self._lora_state_dict(self.get_bare_model(self.net_g))}
        if hasattr(self, 'net_g_ema'):
            save_dict['params_ema'] = self._lora_state_dict(self.get_bare_model(self.net_g_ema))

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
        self._save_lora_network(current_iter)
        self.save_training_state(epoch, current_iter)
