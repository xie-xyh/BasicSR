import argparse
import json
from pathlib import Path

from compare_paired_tables import compare, load_arm


def rank(protocol):
    width = protocol / 'paired_width_8000/pairs.csv'
    pyramid = protocol / 'paired_8000/pairs.csv'
    gate = protocol / 'paired_gate_8000/pairs.csv'
    control = load_arm(width, 'control')
    comparisons = {}
    for name, path, arm in (('011f', width, 'candidate'), ('011d_multi', pyramid, 'candidate'),
                            ('011d_single', pyramid, 'control'), ('011g', gate, 'candidate')):
        comparisons[name] = compare(control, load_arm(path, arm))

    rows = []
    for name, report in comparisons.items():
        dense = report['OWH-Dense-internal-v2']
        mist = report['OWH-Mist-internal-v2']
        psnr = [dense['psnr']['delta'], mist['psnr']['delta']]
        ssim = [dense['ssim']['delta'], mist['ssim']['delta']]
        rows.append(dict(name=name, min_psnr_gain=min(psnr), mean_psnr_gain=sum(psnr) / 2,
                         eligible=min(psnr) >= -0.10 and max(psnr) >= 0.10 and min(ssim) >= -0.001,
                         target_deltas=dict(Dense=dict(psnr=psnr[0], ssim=ssim[0]),
                                            Mist=dict(psnr=psnr[1], ssim=ssim[1]))))
    rows.sort(key=lambda row: (row['eligible'], row['min_psnr_gain'], row['mean_psnr_gain']), reverse=True)
    return dict(nomination=rows[0]['name'] if rows[0]['eligible'] else None,
                ranking=rows, comparisons=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = rank(args.protocol)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(nomination=result['nomination'], ranking=result['ranking']), indent=2))
