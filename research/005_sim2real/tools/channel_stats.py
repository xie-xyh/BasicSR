# Channel statistics analysis for the sim2real gap: real haze (OWH real-test, no GT)
# vs synthetic haze (OWH Dense/Mist/Moderate test hazy) vs clean GT.
# Per image: per-channel means, channel dispersion (max-min), dominant channel,
# atmospheric-light color estimate (top-0.1% intensity pixels), brightness.
# Usage: python research/005_sim2real/tools/channel_stats.py

import os
import csv

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = 'research/005_sim2real'

DATASETS = {
    'real-test': ('datasets/OWH/real-test', '.jpeg'),
    'dense-hazy': ('datasets/OWH/Dense/test/hazy', '.jpg'),
    'mist-hazy': ('datasets/OWH/Mist/test/hazy', '.jpg'),
    'moderate-hazy': ('datasets/OWH/Moderate/test/hazy', '.jpg'),
    'GT (dense clear)': ('datasets/OWH/Dense/test/clear', '.jpg'),
}


def stats_of(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = img[:, :, ::-1].astype(np.float64) / 255.0  # RGB
    ch_mean = img.mean(axis=(0, 1))  # R, G, B
    disp = (img.max(axis=2) - img.min(axis=2)).mean()
    intensity = img.mean(axis=2)
    thr = np.percentile(intensity, 99.9)
    A = img[intensity >= thr].mean(axis=0)  # atmospheric light color estimate
    return ch_mean, disp, A, intensity.mean(), img.shape


def main():
    rows = {}
    for name, (root, ext) in DATASETS.items():
        files = sorted(f for f in os.listdir(root) if f.endswith(ext))
        recs = []
        for f in files:
            s = stats_of(os.path.join(root, f))
            if s is None:
                continue
            ch_mean, disp, A, bright, shape = s
            recs.append(dict(file=f, R=ch_mean[0], G=ch_mean[1], B=ch_mean[2],
                             dispersion=disp, A_R=A[0], A_G=A[1], A_B=A[2],
                             brightness=bright, h=shape[0], w=shape[1],
                             dom='RGB'[int(np.argmax(ch_mean))]))
        rows[name] = recs
        print(f'{name}: {len(recs)} imgs, size range '
              f'{min(r["h"] for r in recs)}x{min(r["w"] for r in recs)}'
              f'~{max(r["h"] for r in recs)}x{max(r["w"] for r in recs)}')

    # save csv
    with open(os.path.join(OUT_DIR, 'channel_stats.csv'), 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['dataset', 'file', 'R', 'G', 'B', 'dispersion', 'A_R', 'A_G', 'A_B',
                       'brightness', 'dom_channel'])
        for name, recs in rows.items():
            for r in recs:
                wcsv.writerow([name, r['file'], f"{r['R']:.4f}", f"{r['G']:.4f}", f"{r['B']:.4f}",
                               f"{r['dispersion']:.4f}", f"{r['A_R']:.4f}", f"{r['A_G']:.4f}",
                               f"{r['A_B']:.4f}", f"{r['brightness']:.4f}", r['dom']])

    colors = {'real-test': 'tab:red', 'dense-hazy': 'tab:blue', 'mist-hazy': 'tab:green',
              'moderate-hazy': 'tab:orange', 'GT (dense clear)': 'tab:gray'}

    # fig 1: channel dispersion distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, recs in rows.items():
        v = [r['dispersion'] for r in recs]
        ax.hist(v, bins=50, density=True, histtype='step', lw=1.8, color=colors[name],
                label=f'{name} (mean {np.mean(v):.3f})')
    ax.set_xlabel('channel dispersion (mean of max-min over RGB)')
    ax.set_ylabel('density')
    ax.set_title('Channel dispersion: real vs synthetic haze')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'channel_dispersion.png'), dpi=150)
    plt.close(fig)

    # fig 2: atmospheric-light chromaticity scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, recs in rows.items():
        A = np.array([[r['A_R'], r['A_G'], r['A_B']] for r in recs])
        s = A.sum(axis=1, keepdims=True)
        chrom = A / np.maximum(s, 1e-8)
        ax.scatter(chrom[:, 0], chrom[:, 2], s=6, alpha=0.5, color=colors[name], label=name)
    ax.plot([0.30, 0.36], [0.30, 0.36], 'k--', lw=0.8)
    ax.scatter([1/3], [1/3], marker='+', c='k', s=80, label='neutral (gray/white haze)')
    ax.set_xlabel('A chromaticity: r/(r+g+b)')
    ax.set_ylabel('A chromaticity: b/(r+g+b)')
    ax.set_title('Estimated atmospheric-light color (top-0.1% intensity pixels)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'airlight_chromaticity.png'), dpi=150)
    plt.close(fig)

    # fig 3: per-channel offset (channel-selective cast) boxplot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = list(rows.keys())
    gmr = [[r['G'] - r['R'] for r in rows[n]] for n in names]
    bmr = [[r['B'] - r['R'] for r in rows[n]] for n in names]
    axes[0].boxplot(gmr, tick_labels=[n.replace(' (dense clear)', '') for n in names], showfliers=False)
    axes[0].axhline(0, c='k', ls=':', lw=0.8)
    axes[0].set_title('G - R channel mean offset')
    axes[1].boxplot(bmr, tick_labels=[n.replace(' (dense clear)', '') for n in names], showfliers=False)
    axes[1].axhline(0, c='k', ls=':', lw=0.8)
    axes[1].set_title('B - R channel mean offset')
    for ax in axes:
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel('offset (0-1 scale)')
    fig.suptitle('Channel-selective color cast: real vs synthetic haze')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'channel_cast.png'), dpi=150)
    plt.close(fig)

    # summary table
    print('\n=== summary ===')
    print(f'{"dataset":<20}{"dispersion":>12}{"B-R mean":>10}{"G-R mean":>10}{"A chrom r":>10}{"A chrom b":>10}{"dom R/G/B %":>16}')
    for name, recs in rows.items():
        disp = np.mean([r['dispersion'] for r in recs])
        bmr_m = np.mean([r['B'] - r['R'] for r in recs])
        gmr_m = np.mean([r['G'] - r['R'] for r in recs])
        A = np.array([[r['A_R'], r['A_G'], r['A_B']] for r in recs])
        chrom = (A / A.sum(axis=1, keepdims=True)).mean(axis=0)
        doms = [r['dom'] for r in recs]
        dom_s = f"{doms.count('R')/len(doms)*100:.0f}/{doms.count('G')/len(doms)*100:.0f}/{doms.count('B')/len(doms)*100:.0f}"
        print(f'{name:<20}{disp:>12.4f}{bmr_m:>+10.4f}{gmr_m:>+10.4f}{chrom[0]:>10.4f}{chrom[2]:>10.4f}{dom_s:>16}')


if __name__ == '__main__':
    main()
