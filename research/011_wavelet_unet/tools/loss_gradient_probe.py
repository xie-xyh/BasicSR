import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np
import torch
import yaml

from basicsr.data import build_dataset
from basicsr.losses import build_loss
from paired_eval import load_network, predict_float


def spectrum(value):
    return torch.fft.fft2(value, norm='ortho')


def measure(output, target, losses):
    error = spectrum((output - target).detach())
    height, width = output.shape[-2:]
    radius = torch.hypot(2 * torch.fft.fftfreq(height)[:, None], 2 * torch.fft.fftfreq(width)[None, :])
    bands = {'low': radius < .02, 'mid': (radius >= .02) & (radius < .15), 'high': radius >= .15}
    metrics = {}
    for name, loss in losses.items():
        gradient, = torch.autograd.grad(loss(output, target), output)
        grad = spectrum(gradient)
        total_energy = grad.abs().square().sum()
        for band, mask in bands.items():
            actual = grad[..., mask]
            expected = error[..., mask]
            energy = actual.abs().square().sum()
            projected = (actual.conj() * expected).real.sum()
            norm = (energy * expected.abs().square().sum()).sqrt()
            metrics[f'{name}_{band}_gradient_share'] = float(energy / total_energy.clamp_min(1e-30))
            metrics[f'{name}_{band}_alignment'] = float(projected / norm.clamp_min(1e-30))
    return metrics


def run(args):
    config = yaml.safe_load(args.config.read_text())
    net = load_network(config, args.weights, 'cpu')
    recipe = config['train']
    losses = {key: build_loss(recipe[key]) for key in ('pixel_opt', 'fft_opt', 'gradient_opt')}

    def subband(output, target):
        options = recipe['subband_opt']
        low, high = net._split_subbands(net.dwt1(output))
        low_gt, high_gt = net._split_subbands(net.dwt1(target))
        return options['loss_weight'] * (options['w_base'] * (low-low_gt).abs().mean()
                                         + options['w_detail'] * (high-high_gt).abs().mean())

    losses['subband'] = subband
    result = {}
    for phase, options in config['datasets'].items():
        if not phase.startswith('val'):
            continue
        data = build_dataset(dict(options, phase='val', scale=1))
        values = defaultdict(list)
        for index in np.linspace(0, len(data)-1, args.limit, dtype=int):
            sample = data[index]
            with torch.no_grad():
                prediction = predict_float(net, sample['lq'])
            prediction.requires_grad_(True)
            for key, value in measure(prediction, sample['gt'][None], losses).items():
                values[key].append(value)
        result[options['name']] = {key: fmean(items) for key, items in values.items()}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'weights', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--limit', type=int, default=8)
    run(parser.parse_args())
