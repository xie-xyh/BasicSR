from os import path as osp

import numpy as np
import torch

from basicsr.utils import scandir
from basicsr.utils.registry import DATASET_REGISTRY

from .paired_image_dataset import PairedImageDataset


@DATASET_REGISTRY.register()
class PairedLogitDataset(PairedImageDataset):
    """Paired restoration images with one precomputed routing vector per LQ image."""

    def __init__(self, opt):
        self.logit_folder = osp.expanduser(opt['dataroot_logit'])
        super().__init__(opt)
        logit_paths = scandir(self.logit_folder, suffix='npy', full_path=True)
        self.logit_paths = {osp.splitext(osp.basename(path))[0]: path for path in logit_paths}
        missing = [
            path['lq_path'] for path in self.paths
            if osp.splitext(osp.basename(path['lq_path']))[0] not in self.logit_paths
        ]
        if missing:
            raise ValueError(f'Missing routing logits for {len(missing)} images, for example: {missing[0]}')

    def __getitem__(self, index):
        data = super().__getitem__(index)
        basename = osp.splitext(osp.basename(data['lq_path']))[0]
        logit_path = self.logit_paths[basename]
        logit = np.load(logit_path).astype(np.float32)
        if logit.ndim != 1:
            raise ValueError(f'Expected a one-dimensional routing vector at {logit_path}, got {logit.shape}.')
        data['logit'] = torch.from_numpy(logit)
        data['logit_path'] = logit_path
        return data
