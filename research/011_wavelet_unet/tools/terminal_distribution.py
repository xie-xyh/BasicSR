import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from audit_final import audit


def analyze(run):
    verified = audit(run)
    rows = defaultdict(list)
    with (run / 'official_60000/pairs.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            scores = [float(row[f'{arm}_{metric}']) for arm in ('control', 'candidate')
                      for metric in ('psnr', 'ssim')]
            rows[row['dataset']].append(scores)
    report = {}
    for split, pairs in rows.items():
        psnr_gains = [candidate - control for control, _, candidate, _ in pairs]
        ssim_gains = [candidate - control for _, control, _, candidate in pairs]
        target = verified[split]['target_db']
        report[split] = {
            **verified[split],
            'both_improved': sum(psnr > 0 and ssim > 0 for psnr, ssim in zip(psnr_gains, ssim_gains)),
            'psnr_drops_over_one_db': sum(gain < -1 for gain in psnr_gains),
            'median_psnr_gain': median(psnr_gains),
            'median_ssim_gain': median(ssim_gains),
            'gap_to_target_db': None if target is None else max(0, target - verified[split]['psnr']['candidate']),
        }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    with (args.run / 'terminal_distribution.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
