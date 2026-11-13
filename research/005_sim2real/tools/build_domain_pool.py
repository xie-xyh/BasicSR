# Build the domain-matched real-haze training pool (Pool A) from OverwaterHaze-ACCV2020.
# Steps: collect train/hazy + natural_test -> exclude ORB>=20 leakage candidates vs
# real-test (from probe/orb_verify.npy) -> drop near-clear images (dark-channel floor)
# -> hold out 400 val -> symlink tree at datasets/OWH-RealPoolA/{train,val}.
# Usage: python research/005_sim2real/tools/build_domain_pool.py

import os
import csv

import cv2
import numpy as np

SRC = '/home/yxy/xyy/datasets/OverwaterHaze-ACCV2020'
OUT = 'datasets/OWH-RealPoolA'
DARK_TH = 0.20        # 暗通道均值下限（清晰图通常 <0.15，留安全余量）
N_VAL = 400
SEED = 0

ORB_NPY = 'research/005_sim2real/probe/orb_verify.npy'


def dark_channel_mean(img):
    return float(img.min(axis=2).mean())


def main():
    # 泄漏剔除名单
    excl = set()
    if os.path.exists(ORB_NPY):
        pairs = np.load(ORB_NPY, allow_pickle=True)
        for pool_path, rt_name, score in pairs:
            if float(score) >= 20:
                excl.add(str(pool_path))
    print(f'ORB>=20 剔除名单: {len(excl)} 张')

    files = []
    for sub in ('train/hazy', 'test/natural_test'):
        subdir = os.path.join(SRC, sub)
        files += sorted(os.path.join(subdir, x) for x in os.listdir(subdir) if x.endswith('.jpg'))
    print(f'原始候选: {len(files)}')

    kept, dropped_leak, dropped_clear = [], 0, 0
    report = []
    for f in files:
        if f in excl:
            dropped_leak += 1
            continue
        img = cv2.imread(f)
        if img is None:
            dropped_clear += 1
            continue
        dc = dark_channel_mean(img.astype(np.float64) / 255.0)
        if dc < DARK_TH:
            dropped_clear += 1
            continue
        kept.append((f, dc))
        report.append((os.path.relpath(f, SRC), f'{dc:.4f}'))
    print(f'剔除泄漏 {dropped_leak}，剔除过清/损坏 {dropped_clear}，保留 {len(kept)}')

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(kept))
    val_idx = set(idx[:N_VAL].tolist())

    for split in ('train', 'val'):
        os.makedirs(os.path.join(OUT, split), exist_ok=True)
    n_train = n_val = 0
    for i, (f, dc) in enumerate(kept):
        split = 'val' if i in val_idx else 'train'
        dst = os.path.join(OUT, split, os.path.basename(f))
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(f), dst)
        if split == 'val':
            n_val += 1
        else:
            n_train += 1
    print(f'Pool A 建成: train {n_train} / val {n_val} -> {OUT}/')

    with open('research/005_sim2real/pool_a_report.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['relpath', 'dark_channel_mean'])
        w.writerows(report)


if __name__ == '__main__':
    main()
