import argparse
import json
from datetime import datetime
from pathlib import Path

from watch_training import ITER, parse


def timeline(log):
    steps = {}
    started = None
    for line in log.read_text().splitlines():
        if 'Start training from epoch:' in line:
            started = timestamp(line)
        match = ITER.search(line)
        if match and started is not None:
            step = int(match[1].replace(',', ''))
            steps[step] = (timestamp(line) - started).total_seconds()
    _, scores = parse(log)
    return [{'step': step, 'elapsed_seconds': steps[step],
             'scores': {dataset: value for (iteration, dataset), value in scores.items() if iteration == step}}
            for step in sorted({iteration for iteration, _ in scores} & steps.keys())]


def timestamp(line):
    return datetime.strptime(line[:23], '%Y-%m-%d %H:%M:%S,%f')


def compare(control, candidate):
    return [{'wavelet': left, 'dea_latest_available': max(available, key=lambda row: row['step']) if available else None}
            for left in control
            for available in [[right for right in candidate if right['elapsed_seconds'] <= left['elapsed_seconds']]]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('backbone_directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    curves = {arm: timeline(args.backbone_directory / f'run_{arm}/console.log') for arm in ('wavelet', 'dea')}
    result = dict(curves=curves, at_wavelet_checkpoints=compare(curves['wavelet'], curves['dea']))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
