import argparse
import csv
from pathlib import Path

import numpy as np
import cv2
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.utils import tensor2img
from paired_eval import load_network, predict


def scores(prediction, target, split_components=False):
    image = prediction.astype(np.float64) / 255
    clear = target.astype(np.float64) / 255
    height, width = image.shape[:2]
    vertical = np.broadcast_to(np.linspace(-1, 1, height)[:, None], (height, width))
    target_flat = clear.reshape(-1, 3)
    result = {}
    for variant in ('original', 'global_affine', 'vertical_affine'):
        adjusted = image
        if variant != 'original':
            base = [image, np.ones_like(image[..., :1])]
            if variant == 'vertical_affine':
                base += [image * vertical[..., None], vertical[..., None]]
            design = np.concatenate(base, axis=-1).reshape(height * width, -1)
            coefficient = np.linalg.lstsq(design[::16], target_flat[::16], rcond=None)[0]
            adjusted = (design @ coefficient).reshape(image.shape).clip(0, 1)
        mse = np.square(adjusted - clear).mean()
        result[variant] = -10 * np.log10(max(mse, np.finfo(float).tiny))
    if split_components:
        smooth = cv2.GaussianBlur(image, (0, 0), 8)
        clear_smooth = cv2.GaussianBlur(clear, (0, 0), 8)
        detail = image - smooth
        target_detail = clear - clear_smooth
        for variant, corrected in (
                ('oracle_smooth_only', clear_smooth + detail),
                ('oracle_detail_only', smooth + target_detail)):
            mse = np.square(corrected.clip(0, 1) - clear).mean()
            result[variant] = -10 * np.log10(max(mse, np.finfo(float).tiny))
    return result


@torch.inference_mode()
def main(args):
    config = yaml.safe_load(args.config.read_text())
    net = load_network(config, args.weights, 'cpu')
    rows = []
    for phase, options in config['datasets'].items():
        if not phase.startswith('val'):
            continue
        dataset = build_dataset(dict(options, phase='val', scale=1))
        count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        for index in np.linspace(0, len(dataset) - 1, count, dtype=int):
            item = dataset[index]
            restored = predict(net, item['lq'])
            rows.append(dict(dataset=options['name'], image=Path(item['gt_path']).name,
                             **scores(restored, tensor2img(item['gt']), args.split_components)))
        print(options['name'], count, flush=True)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for subset in sorted({row['dataset'] for row in rows}):
        group = [row for row in rows if row['dataset'] == subset]
        print(subset, {key: round(float(np.mean([row[key] for row in group])), 4)
                       for key in group[0] if key not in ('dataset', 'image')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'weights', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--split-components', action='store_true')
    cv2.setNumThreads(1)
    main(parser.parse_args())
