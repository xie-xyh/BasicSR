import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric
from basicsr.utils import tensor2img
from paired_eval import band_mse, load_network, predict, summarize


@torch.inference_mode()
def evaluate(args):
    config = yaml.safe_load(args.config.read_text())
    network = load_network(config, args.weights, args.device)
    names = ('level3_fgb1', 'level3_fgb2') if args.branch == 'dea_lfm_pair' else (args.branch,)
    branches = [getattr(network, name) for name in names]
    if any(branch is None for branch in branches):
        raise ValueError(f'Inactive branch: {args.branch}')
    rows = []
    for phase, options in config['datasets'].items():
        if not phase.startswith('val'):
            continue
        data = build_dataset(dict(options, phase='val', scale=1))
        total = len(data) if args.limit is None else min(len(data), args.limit)
        for index in np.linspace(0, len(data) - 1, total, dtype=int):
            item = data[index]
            image, target = item['lq'].to(args.device), tensor2img(item['gt'])
            gt_path = Path(item['gt_path'])
            row = dict(dataset=options['name'], image=gt_path.name,
                       gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest())
            def bypass(module, inputs, output):
                return inputs[0] if args.identity else torch.zeros_like(output)

            with ExitStack() as hooks:
                for branch in branches:
                    hooks.enter_context(branch.register_forward_hook(bypass))
                control = predict(network, image)
            candidate = predict(network, image)
            for arm, output in (('control', control), ('candidate', candidate)):
                metrics = band_mse(output, target)
                for name in ('psnr', 'ssim'):
                    options_metric = dict(type=f'calculate_{name}', crop_border=0, test_y_channel=False)
                    metrics[name] = calculate_metric(dict(img=output, img2=target), options_metric)
                row.update({f'{arm}_{name}': value for name, value in metrics.items()})
            rows.append(row)
        print(f'{options["name"]}: {total} branch-toggle pairs', flush=True)
    result = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'weights', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--branch', choices=('coarse_color_residual', 'mid_band_residual',
                                             'lfm', 'level3_fgb1', 'level3_fgb2', 'dea_lfm_pair'),
                        default='coarse_color_residual')
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.branch in ('lfm', 'level3_fgb1', 'level3_fgb2', 'dea_lfm_pair') and not args.identity:
        parser.error('Residual modulation blocks require --identity.')
    evaluate(args)
