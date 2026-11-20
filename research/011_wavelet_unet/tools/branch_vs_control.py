import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import fmean


def load(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    table = {(row['dataset'], row['image'], row['gt_sha256']): row for row in rows}
    if not rows or len(table) != len(rows):
        raise ValueError(f'Empty or duplicate image keys: {path}')
    return table


def compare(independent, toggle):
    if independent.keys() != toggle.keys():
        raise ValueError('Image coverage differs between reports')
    splits = defaultdict(list)
    for key, row in independent.items():
        toggled = toggle[key]
        enabled = float(row['candidate_psnr'])
        if abs(enabled - float(toggled['candidate_psnr'])) > 0.03:
            raise ValueError(f'Enabled candidate mismatch: {key}')
        splits[key[0]].append((enabled - float(row['control_psnr']),
                               float(toggled['candidate_psnr']) - float(toggled['control_psnr'])))
    return {
        split: dict(images=len(values), independent_gain_db=fmean(net for net, on_off in values),
                    toggle_gain_db=fmean(on_off for net, on_off in values),
                    helped_both=sum(net > 0 and on_off > 0 for net, on_off in values),
                    branch_used_but_no_net_gain=sum(net <= 0 and on_off > 0 for net, on_off in values))
        for split, values in sorted(splits.items())
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('independent', 'toggle', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    report = compare(load(args.independent), load(args.toggle))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, indent=2, allow_nan=False))
