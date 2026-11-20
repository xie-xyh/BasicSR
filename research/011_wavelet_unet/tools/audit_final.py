import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean

import yaml


EXPECTED = {'OWH-Dense-test': (905, 27), 'OWH-Mist-test': (892, 35),
            'OWH-Moderate-test': (914, None)}


def verify_images(rows, clear_dir, hazy_dir):
    names = {row['image'] for row in rows}
    for folder in (clear_dir, hazy_dir):
        files = {item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')}
        if files != names:
            raise ValueError(f'Official image coverage differs in {folder}')
    for row in rows:
        digest = hashlib.sha256((clear_dir / row['image']).read_bytes()).hexdigest()
        if digest != row['gt_sha256']:
            raise ValueError(f'Ground truth changed: {clear_dir / row["image"]}')


def audit(directory):
    for arm in ('control', 'candidate'):
        logs = directory / 'run'
        if (logs / f'{arm}.exit').read_text().strip() != '0':
            raise ValueError(f'{arm} did not exit cleanly')
        steps = re.findall(r'iter:\s*([\d,]+), lr:', (logs / f'{arm}.log').read_text())
        if not steps or int(steps[-1].replace(',', '')) != 60000:
            raise ValueError(f'{arm} did not end at 60000 updates')

    scores = directory / 'official_60000'
    summary = json.loads((scores / 'summary.json').read_text())
    splits = defaultdict(list)
    with (scores / 'pairs.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            splits[row['dataset']].append(row)
    if set(splits) != set(EXPECTED) or set(summary) != set(EXPECTED):
        raise ValueError('Unexpected official test splits')
    control_config, candidate_config = [
        yaml.safe_load((directory / f'{arm}_official.yml').read_text())
        for arm in ('control', 'candidate')
    ]
    if control_config['datasets'] != candidate_config['datasets']:
        raise ValueError('Official evaluation configurations refer to different data.')
    datasets = {value['name']: value for name, value in control_config['datasets'].items()
                if name.startswith('val')}
    if set(datasets) != set(EXPECTED):
        raise ValueError('Unexpected official evaluation configuration')
    report = {}
    for split, (count, target) in EXPECTED.items():
        rows = splits[split]
        if len(rows) != count or len({row['image'] for row in rows}) != count or summary[split]['n'] != count:
            raise ValueError(f'Incorrect image coverage for {split}')
        options = datasets[split]
        verify_images(rows, Path(options['dataroot_gt']), Path(options['dataroot_lq']))
        metrics = {}
        for metric in ('psnr', 'ssim'):
            control, candidate = (fmean(float(row[f'{arm}_{metric}']) for row in rows)
                                  for arm in ('control', 'candidate'))
            delta = candidate - control
            measured = dict(control=control, candidate=candidate, delta=delta)
            for key, value in measured.items():
                if not math.isfinite(value) or not math.isclose(value, summary[split][metric][key], abs_tol=1e-9):
                    raise ValueError(f'Summary mismatch for {split}: {metric}/{key}')
            measured['ci95'] = summary[split][metric]['ci95']
            metrics[metric] = measured
        report[split] = dict(images=count, **metrics, target_db=target,
                             reached=None if target is None else metrics['psnr']['candidate'] >= target)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = audit(args.run)
    with (args.run / 'official_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))
