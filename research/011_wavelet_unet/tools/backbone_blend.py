import argparse
import csv
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric
from basicsr.utils import tensor2img
from paired_eval import load_network, predict_float


@torch.inference_mode()
def evaluate(folder, limit):
    configs = [yaml.safe_load((folder / f'{name}.yml').read_text()) for name in ('wavelet', 'dea')]
    if configs[0]['datasets'] != configs[1]['datasets']:
        raise ValueError('Backbones must use the same datasets.')
    networks = [load_network(cfg, Path('experiments') / cfg['name'] / 'models/net_g_latest.pth', 'cpu')
                for cfg in configs]
    rows = []
    for phase, options in configs[0]['datasets'].items():
        if not phase.startswith('val'):
            continue
        data = build_dataset(dict(options, phase='val', scale=1))
        block = []
        for index in np.linspace(0, len(data) - 1, min(len(data), limit), dtype=int):
            item = data[index]
            wavelet, dea = (predict_float(net, item['lq']).clamp(0, 1) for net in networks)
            target = tensor2img(item['gt'])
            row = dict(dataset=options['name'], image=Path(item['gt_path']).name)
            for label, output in (('wavelet', wavelet), ('dea', dea), ('average', (wavelet + dea) / 2)):
                prediction = tensor2img(output)
                for metric in ('psnr', 'ssim'):
                    row[f'{label}_{metric}'] = calculate_metric(
                        dict(img=prediction, img2=target), dict(type=f'calculate_{metric}', crop_border=0))
            block.append(row)
        print(options['name'], {key: fmean(row[key] for row in block)
                               for key in block[0] if key not in ('dataset', 'image')}, flush=True)
        rows.extend(block)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--limit', type=int, default=8)
    args = parser.parse_args()
    records = evaluate(args.folder, args.limit)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
