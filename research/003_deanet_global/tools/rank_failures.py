"""Rank per-image PSNR/SSIM of a finished eval to locate failure cases.

Usage:
    python research/003_deanet_global/tools/rank_failures.py \
        --result results/001_dense_eval/visualization/OWH-Dense-test \
        --gt datasets/OWH/Dense/test/clear \
        --lq datasets/OWH/Dense/test/hazy \
        --out research/003_deanet_global/failure_cases/001_dense_psnr.csv
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True)
    p.add_argument('--gt', required=True)
    p.add_argument('--lq', default=None, help='hazy dir; also reports hazy-vs-GT PSNR for reference')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    rows = []
    for f in sorted(os.listdir(args.result)):
        name = re.match(r'(\d+)', f).group(1)
        out = cv2.imread(os.path.join(args.result, f))
        gt_path = None
        for ext in ('.png', '.jpg', '.jpeg'):
            cand = os.path.join(args.gt, name + ext)
            if os.path.exists(cand):
                gt_path = cand
                break
        if gt_path is None or out is None:
            continue
        gt = cv2.imread(gt_path)
        if gt.shape != out.shape:
            h = min(gt.shape[0], out.shape[0])
            w = min(gt.shape[1], out.shape[1])
            gt, out = gt[:h, :w], out[:h, :w]
        row = {'name': name, 'psnr_out': psnr(out, gt)}
        if args.lq:
            for ext in ('.png', '.jpg', '.jpeg'):
                cand = os.path.join(args.lq, name + ext)
                if os.path.exists(cand):
                    lq = cv2.imread(cand)[:gt.shape[0], :gt.shape[1]]
                    row['psnr_lq'] = psnr(lq, gt)
                    break
        rows.append(row)

    rows.sort(key=lambda r: r['psnr_out'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['name', 'psnr_out', 'psnr_lq'])
        writer.writeheader()
        for r in rows:
            r.setdefault('psnr_lq', '')
            writer.writerow(r)

    ps = np.array([r['psnr_out'] for r in rows])
    lq_ps = np.array([r['psnr_lq'] for r in rows if r['psnr_lq'] != ''])
    print(f'total: {len(rows)}  mean PSNR out: {ps.mean():.4f}  (hazy baseline: {lq_ps.mean():.4f})')
    print(f'min: {ps.min():.2f}  p5: {np.percentile(ps, 5):.2f}  p25: {np.percentile(ps, 25):.2f}  '
          f'median: {np.median(ps):.2f}')
    print('\nworst 15:')
    for r in rows[:15]:
        print(f"  {r['name']}  out={r['psnr_out']:.2f}  hazy={r['psnr_lq'] if r['psnr_lq']=='' else format(r['psnr_lq'], '.2f')}")


if __name__ == '__main__':
    main()
