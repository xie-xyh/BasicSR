import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from paired_eval import load_network, predict_float


def radius_at_fraction(gradient, row, col, fraction):
    influence = gradient.abs().sum(dim=0)
    height, width = influence.shape
    vertical = torch.arange(height, dtype=influence.dtype)[:, None] - row
    horizontal = torch.arange(width, dtype=influence.dtype)[None, :] - col
    radius = torch.hypot(vertical, horizontal)
    distance = radius.flatten().argsort()
    mass = influence.flatten()[distance].cumsum(0)
    cutoff = torch.searchsorted(mass, mass[-1] * fraction)
    return radius.flatten()[distance[cutoff]].item()


def probe(net, image):
    image = image.detach().requires_grad_(True)
    result = predict_float(net, image)[0]
    height, width = result.shape[-2:]
    col = width // 2
    rows = {}
    for index, row in enumerate((height // 4, height // 2, 3 * height // 4)):
        region = result[:, row - 8:row + 9, col - 8:col + 9]
        gradient, = torch.autograd.grad(region.mean(), image, retain_graph=index < 2)
        rows[('upper', 'center', 'lower')[index]] = radius_at_fraction(gradient, row, col, .9)
    return rows


def run(args):
    config = yaml.safe_load(args.config.read_text())
    net = load_network(config, args.weights, 'cpu').requires_grad_(False)
    result = {}
    for phase, options in config['datasets'].items():
        if not phase.startswith('val'):
            continue
        dataset = build_dataset(dict(options, phase='val', scale=1))
        rows = []
        for index in np.linspace(0, len(dataset) - 1, min(args.limit, len(dataset)), dtype=int):
            item = dataset[index]
            rows.append(dict(image=Path(item['lq_path']).name, **probe(net, item['lq'])))
        medians = {region: median(row[region] for row in rows) for region in ('upper', 'center', 'lower')}
        result[options['name']] = dict(median_r90_pixels=medians, samples=rows)
        print(options['name'], medians, flush=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'weights', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--limit', type=int, default=4)
    run(parser.parse_args())
