import argparse
import json
import time
from pathlib import Path
from statistics import median

import torch
from torch import nn
import yaml

from basicsr.archs.deanet_arch import DEConv
from paired_eval import load_network, predict_float


@torch.no_grad()
def fold(module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, DEConv):
            weight, bias = child.get_merged_weight()
            plain = nn.Conv2d(weight.size(1), weight.size(0), 3, padding=1).to(weight)
            plain.weight.copy_(weight)
            plain.bias.copy_(bias)
            setattr(module, name, plain.eval())
            count += 1
        else:
            count += fold(child)
    return count


@torch.inference_mode()
def benchmark(network, sample):
    predict_float(network, sample)
    times = []
    for _ in range(5):
        started = time.perf_counter()
        predict_float(network, sample)
        times.append(time.perf_counter() - started)
    return median(times)


@torch.inference_mode()
def verify(config, weights):
    net = load_network(config, weights, 'cpu')
    generator = torch.Generator().manual_seed(0)
    samples = [torch.rand(3, height, width, generator=generator)
               for height, width in ((64, 64), (65, 81), (256, 384))]
    references = [predict_float(net, item) for item in samples]
    before = benchmark(net, samples[-1])
    count = fold(net)
    if count == 0:
        raise ValueError('No DEConv blocks found.')
    max_difference = 0.
    for item, expected in zip(samples, references):
        actual = predict_float(net, item)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        max_difference = max(max_difference, (actual - expected).abs().max().item())
    after = benchmark(net, samples[-1])
    return dict(folded_layers=count, max_difference=max_difference,
                median_seconds_before=before, median_seconds_after=after,
                timed_shape=list(samples[-1].shape))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('config', 'weights', 'output'):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    report = verify(yaml.safe_load(args.config.read_text()), args.weights)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(report)
