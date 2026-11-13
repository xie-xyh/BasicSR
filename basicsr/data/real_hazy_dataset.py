# RealHazyDataset: unpaired real hazy images for self-supervised adapter training.
# Returns {'lq', 'lq_path'} with random crop + flip/rotation augmentation; no GT.

import numpy as np
from torch.utils import data as data

from basicsr.data.data_util import paths_from_lmdb
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RealHazyDataset(data.Dataset):
    """Unpaired real hazy images with random-crop augmentation (adapter training).

    Args:
        opt (dict): dataroot_lq (str), crop_size (int), use_hflip (bool),
            use_rot (bool), io_backend (dict). No GT is read.
    """

    def __init__(self, opt):
        super(RealHazyDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.lq_folder = opt['dataroot_lq']
        self.crop_size = opt.get('crop_size', 256)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.lq_folder)
        else:
            self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img = imfrombytes(img_bytes, float32=True)

        # random crop to crop_size (upscale short side if the image is smaller)
        h, w = img.shape[:2]
        s = self.crop_size
        if min(h, w) < s:
            scale = s / min(h, w)
            import cv2
            img = cv2.resize(img, (int(np.ceil(w * scale)), int(np.ceil(h * scale))),
                             interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        top = np.random.randint(0, h - s + 1)
        left = np.random.randint(0, w - s + 1)
        img = img[top:top + s, left:left + s]

        img = augment(img, self.opt.get('use_hflip', True), self.opt.get('use_rot', True))
        return {'lq': img2tensor(img, bgr2rgb=True, float32=True), 'lq_path': lq_path}

    def __len__(self):
        return len(self.paths)
