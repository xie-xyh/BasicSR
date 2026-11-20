import argparse
import hashlib
from pathlib import Path

import torch


def verify(source, calibrated):
    original = torch.load(source, map_location='cpu')['params_ema']
    updated = torch.load(calibrated, map_location='cpu')['params_ema']
    missing = set(original) - set(updated)
    modified = [key for key, value in original.items()
                if key in updated and not torch.equal(value, updated[key])]
    additions = set(updated) - set(original)
    if missing or modified or not additions or any(not key.startswith('color_head.') for key in additions):
        raise ValueError(f'Frozen-weight mismatch: missing={missing}, modified={modified}, added={additions}')
    if not updated['color_head.6.weight'].count_nonzero().item():
        raise ValueError('The color-head output weights stayed at zero.')
    print(f'Verified {len(original)} frozen EMA tensors, {len(additions)} new tensors.')
    for path in (source, calibrated):
        print(hashlib.sha256(path.read_bytes()).hexdigest(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('calibrated', type=Path)
    args = parser.parse_args()
    verify(args.source, args.calibrated)
