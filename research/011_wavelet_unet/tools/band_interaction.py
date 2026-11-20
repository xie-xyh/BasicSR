import argparse
import json
import math
from pathlib import Path

from compare_paired_tables import load_arm
from paired_eval import summarize


def analyze(root):
    low_report = root / 'paired_coarse_8000/pairs.csv'
    mid_report = root / 'paired_mid_8000/pairs.csv'
    combined_report = root / 'paired_combined_vs_mid_8000/pairs.csv'
    plain, low, mid, joint = [load_arm(path, arm) for path, arm in (
        (mid_report, 'control'), (low_report, 'candidate'),
        (mid_report, 'candidate'), (combined_report, 'candidate'))]
    repeated_plain = load_arm(low_report, 'control')
    repeated_mid = load_arm(combined_report, 'control')
    if any(table.keys() != plain.keys() for table in (low, mid, joint, repeated_plain, repeated_mid)):
        raise ValueError('Image keys differ across band experiments.')
    for first, second in ((plain, repeated_plain), (mid, repeated_mid)):
        if any(not math.isclose(first[key]['psnr'], second[key]['psnr'], rel_tol=0, abs_tol=0.03)
               for key in plain):
            raise ValueError('Repeated checkpoint scores do not agree.')
    return interactions(plain, low, mid, joint)


def interactions(plain, low, mid, joint):
    rows = []
    for key in sorted(plain):
        row = dict(zip(('dataset', 'image', 'gt_sha256'), key))
        for metric in plain[key]:
            row[f'control_{metric}'] = low[key][metric] - plain[key][metric]
            row[f'candidate_{metric}'] = joint[key][metric] - mid[key][metric]
        rows.append(row)
    return summarize(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.protocol)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
