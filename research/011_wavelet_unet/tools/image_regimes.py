import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml


def image_features(hazy, clear):
    lq = hazy.astype(np.float32) / 255
    gt = clear.astype(np.float32) / 255
    difference = lq - gt
    input_mse = np.square(difference).mean()
    low_difference = cv2.GaussianBlur(difference, (0, 0), 8)
    low_mse = np.square(low_difference).mean()

    def gradient_power(image):
        return (np.square(np.diff(image, axis=0)).mean()
                + np.square(np.diff(image, axis=1)).mean()) / 2

    detail_ratio = gradient_power(lq) / max(float(gradient_power(gt)), 1e-12)
    height, width = hazy.shape[:2]
    height, width = height // 16 * 16, width // 16 * 16

    def tiles(image):
        return image[:height, :width].reshape(height // 16, 16, width // 16, 16, 3)

    lq_tiles, gt_tiles = tiles(hazy), tiles(gt)
    flat = (lq_tiles.max(axis=(1, 3)) == lq_tiles.min(axis=(1, 3))).all(axis=-1)
    textured = np.var(gt_tiles.astype(np.float32), axis=(1, 3)).mean(axis=-1) >= 9
    return {'input_mse': float(input_mse), 'input_low_mse': float(low_mse),
            'detail_ratio': float(detail_ratio), 'flat_textured_tiles': int((flat & textured).sum())}


def analyze(config, pairs):
    sources = {item['name']: item for key, item in config['datasets'].items() if key.startswith('val')}
    for row in pairs:
        options = sources[row['dataset']]
        name = row['image']
        clear = cv2.imread(str(Path(options['dataroot_gt']) / name))
        hazy = cv2.imread(str(Path(options['dataroot_lq']) / name))
        if clear is None or hazy is None or clear.shape != hazy.shape:
            raise ValueError(f'Invalid image pair: {name}')
        row.update(image_features(hazy, clear))
        for metric in ('psnr', 'ssim', 'low_mse', 'high_mse'):
            row[f'delta_{metric}'] = float(row[f'candidate_{metric}']) - float(row[f'control_{metric}'])
        yield row


def summarize(rows):
    for dataset in sorted({row['dataset'] for row in rows}):
        subset = [row for row in rows if row['dataset'] == dataset]
        saturated = sum(row['flat_textured_tiles'] > 0 for row in subset)
        print(f'{dataset}: {saturated}/{len(subset)} images with flat input / textured target tiles')
        for feature in ('input_low_mse', 'detail_ratio'):
            ordered = sorted(subset, key=lambda row: row[feature])
            for index, group_indices in enumerate(np.array_split(np.arange(len(ordered)), 3), 1):
                group = [ordered[position] for position in group_indices]
                values = [row[feature] for row in group]
                means = {key: np.mean([row[f'delta_{key}'] for row in group])
                         for key in ('psnr', 'ssim', 'low_mse', 'high_mse')}
                print(f'  {feature} tercile {index}: n={len(group)}, {min(values):.4g}–{max(values):.4g}, '
                      + ', '.join(f'Δ{key}={value:+.5g}' for key, value in means.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('pairs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    config = yaml.safe_load(args.config.read_text())
    with args.pairs.open() as stream:
        rows = list(analyze(config, csv.DictReader(stream)))
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)


if __name__ == '__main__':
    main()
