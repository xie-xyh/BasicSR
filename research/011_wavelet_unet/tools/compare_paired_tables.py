import argparse
import csv
import json
from pathlib import Path

from paired_eval import summarize


KEYS = ('dataset', 'image', 'gt_sha256')
METRICS = ('psnr', 'ssim', 'low_mse', 'mid_mse', 'high_mse')


def load_arm(path, arm):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    keyed = {tuple(row[key] for key in KEYS): {metric: float(row[f'{arm}_{metric}']) for metric in METRICS}
             for row in rows}
    if not rows or len(keyed) != len(rows):
        raise ValueError(f'Empty table or repeated image key: {path}')
    return keyed


def compare(control, candidate):
    if control.keys() != candidate.keys():
        raise ValueError('Both tables must contain the same images and target hashes.')
    pairs = []
    for key, metrics in sorted(control.items()):
        row = dict(zip(KEYS, key))
        row.update({f'control_{metric}': value for metric, value in metrics.items()})
        row.update({f'candidate_{metric}': value for metric, value in candidate[key].items()})
        pairs.append(row)
    return summarize(pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--control-arm', choices=('control', 'candidate'), default='control')
    parser.add_argument('--candidate-arm', choices=('control', 'candidate'), default='candidate')
    args = parser.parse_args()
    result = compare(load_arm(args.control, args.control_arm), load_arm(args.candidate, args.candidate_arm))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
