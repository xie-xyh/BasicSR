"""按最小边长过滤成对数据集，生成符号链接副本。

用法：
    python scripts/data_preparation/filter_min_size.py \
        --lq datasets/OWH/Dense/train/hazy --gt datasets/OWH/Dense/train/clear \
        --out datasets/OWH/Dense_ge512/train --min-size 512

输出目录下生成 hazy/ 与 clear/ 两个软链接目录，只保留宽和高都 >= min-size 的配对。
"""

import argparse
import os
import os.path as osp

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lq', required=True, help='lq 图目录')
    parser.add_argument('--gt', required=True, help='gt 图目录')
    parser.add_argument('--out', required=True, help='输出根目录（下含 hazy/ 与 clear/）')
    parser.add_argument('--min-size', type=int, required=True, help='宽高的最小阈值')
    args = parser.parse_args()

    out_lq = osp.join(args.out, 'hazy')
    out_gt = osp.join(args.out, 'clear')
    os.makedirs(out_lq, exist_ok=True)
    os.makedirs(out_gt, exist_ok=True)

    lq_names = sorted(os.listdir(args.lq))
    n_kept, n_drop = 0, 0
    for name in lq_names:
        lq_path = osp.join(args.lq, name)
        gt_path = osp.join(args.gt, name)
        if not osp.exists(gt_path):
            print(f'[skip] gt missing for {name}')
            n_drop += 1
            continue
        w, h = Image.open(lq_path).size
        if w >= args.min_size and h >= args.min_size:
            os.symlink(osp.abspath(lq_path), osp.join(out_lq, name))
            os.symlink(osp.abspath(gt_path), osp.join(out_gt, name))
            n_kept += 1
        else:
            n_drop += 1
    print(f'kept {n_kept}, dropped {n_drop} (min size {args.min_size})')


if __name__ == '__main__':
    main()
