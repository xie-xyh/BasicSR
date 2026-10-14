"""Calculate no-reference metrics (FADE / NIQE / BRISQUE) for all images in a folder.

Usage:
    python scripts/metrics/calculate_nr_metrics.py -i results_folder [--metrics fade niqe brisque]

Requires: pip install fade-python pyiqa

Notes:
- All metrics: the LOWER the better (FADE = fog density, NIQE/BRISQUE = distortion).
- Input order: images are read as RGB; suitable for dehazed outputs of any method.
"""

import argparse
import os
from os import path as osp

import numpy as np
from tqdm import tqdm

from basicsr.metrics import calculate_brisque, calculate_fade, calculate_niqe
from basicsr.utils.img_util import imfrombytes

METRIC_FUNCS = {'fade': calculate_fade, 'niqe': calculate_niqe, 'brisque': calculate_brisque}


def main():
    parser = argparse.ArgumentParser(description='Calculate no-reference quality metrics for a folder of images')
    parser.add_argument('-i', '--input', type=str, required=True, help='folder of images (recursive)')
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['fade', 'niqe', 'brisque'],
        choices=list(METRIC_FUNCS.keys()),
        help='metrics to calculate')
    parser.add_argument('--ext', type=str, default='.png', help='image extension filter')
    args = parser.parse_args()

    suffix = args.ext if args.ext.startswith('.') else f'.{args.ext}'
    paths = []
    for root, _, names in os.walk(args.input):
        for name in names:
            if name.lower().endswith(suffix):
                paths.append(osp.join(root, name))
    paths.sort()
    assert len(paths) > 0, f'No images with suffix {suffix} found under {args.input}'
    print(f'{len(paths)} images found')

    scores = {m: [] for m in args.metrics}
    for p in tqdm(paths, desc='evaluating'):
        with open(p, 'rb') as f:
            img = imfrombytes(f.read(), float32=False)  # HWC uint8 RGB
        for m in args.metrics:
            scores[m].append(METRIC_FUNCS[m](img))

    print('\n===== Results =====')
    for m in args.metrics:
        arr = np.array(scores[m])
        print(f'{m.upper():>8}: mean {arr.mean():.4f} | std {arr.std():.4f} | median {np.median(arr):.4f}')


if __name__ == '__main__':
    main()
