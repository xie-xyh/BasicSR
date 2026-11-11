# Inspect the learned FreqGlobalBlock filters of 003d (DEANetV2).
# Reads the final checkpoint and reports:
#   - magnitude statistics of the complex filter (is it still ~0, i.e. unused?)
#   - radial energy distribution (which frequency bands did it learn to modulate?)
#   - heatmap of the channel-averaged |W| in the canonical rfft layout
# Usage: python research/003d_deanetv2/tools/inspect_fgb_filter.py

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

CKPT = 'experiments/003d_DEANetV2_OWH-Mixed512_L1FFTGrad_60k_B16G1/models/net_g_60000.pth'
OUT_DIR = 'research/003d_deanetv2'


def radial_profile(mag):
    """mag: (h, w) rfft-layout magnitude (row=v, col=u, DC at [0,0]).
    Returns radially averaged profile indexed by integer radius."""
    h, w = mag.shape
    v, u = np.indices((h, w))
    v = np.minimum(v, h - v)  # fftshift-free: full-height layout, v wraps
    r = np.sqrt(u**2 + v**2).astype(np.int32)
    prof = np.zeros(r.max() + 1, dtype=np.float64)
    cnt = np.zeros_like(prof)
    np.add.at(prof, r.ravel(), mag.ravel())
    np.add.at(cnt, r.ravel(), 1)
    return prof / np.maximum(cnt, 1), r


def main():
    sd_all = torch.load(CKPT, map_location='cpu')
    for key in ('params_ema', 'params'):
        if key not in sd_all:
            continue
        sd = sd_all[key]
        print(f'===== param key: {key} =====')
        for blk in ('level3_fgb1', 'level3_fgb2'):
            w = sd[f'{blk}.filter_weight']  # C, h, w', 2
            W = torch.view_as_complex(w.float())  # C, h, w'
            mag = W.abs()  # C, h, w'
            print(f'[{blk}] shape {tuple(W.shape)}')
            print(f'  |W| mean {mag.mean():.6f}  max {mag.max():.6f}  '
                  f'median {mag.median():.6f}  std {mag.std():.6f}')
            # DC component (u=0, v=0) separately
            print(f'  |W| at DC (v=0,u=0): mean {mag[:, 0, 0].mean():.6f}')

            mag_mean = mag.mean(0).numpy()  # h, w'
            prof, r = radial_profile(mag_mean)
            # band energy shares of |W|^2 (energy of the filter response)
            e = (mag_mean**2)
            bands = [(0, 2), (2, 6), (6, 12), (12, mag_mean.shape[0])]
            tot = e.sum()
            shares = []
            for lo, hi in bands:
                mask = (r >= lo) & (r < hi)
                shares.append(e[mask].sum() / tot * 100)
            print('  filter energy share by radius band '
                  '[0,2)/[2,6)/[6,12)/[12,+): ' +
                  ' / '.join(f'{s:.1f}%' for s in shares))
            print(f'  radial |W| profile (r=0..12): ' +
                  ' '.join(f'{prof[i]:.4f}' for i in range(13)))

            # heatmap of channel-mean |W|
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(mag_mean, cmap='viridis', origin='lower')
            ax.set_title(f'{blk} |W| mean over ch ({key})')
            ax.set_xlabel('u (freq x)'); ax.set_ylabel('v (freq y)')
            fig.colorbar(im)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f'fgb_filter_{blk}_{key}.png'), dpi=150)
            plt.close(fig)

        # FFN weight norms for reference (is the FFN branch doing something?)
        for blk in ('level3_fgb1', 'level3_fgb2'):
            w0 = sd[f'{blk}.ffn.0.weight']
            w2 = sd[f'{blk}.ffn.2.weight']
            print(f'[{blk}] ffn conv1 |W| {w0.abs().mean():.4f}, conv2 |W| {w2.abs().mean():.4f}')


if __name__ == '__main__':
    main()
