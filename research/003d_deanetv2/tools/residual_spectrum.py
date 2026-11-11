# Residual (output - GT) radial power-spectrum analysis on OWH Dense-test.
# For each model, accumulate the fraction of residual energy in radial frequency
# bands (radius normalized by the Nyquist of the smaller image dimension),
# averaged per image. Also reports mean PSNR as a sanity check.
# Usage: python research/003d_deanetv2/tools/residual_spectrum.py

import os

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GT_DIR = 'datasets/OWH/Dense/test/clear'
OUT_DIR = 'research/003d_deanetv2'

MODELS = {
    'hazy (input)': ('datasets/OWH/Dense/test/hazy', '{id}.jpg'),
    '001 (L1)': ('results/001_dense_eval/visualization/OWH-Dense-test', '{id}_001_dense_eval.png'),
    '003c (L1+FFT+Grad)': (
        'results/003c_dense_eval/visualization/OWH-Dense-test', '{id}_003c_dense_eval.png'),
    '003d (DEANetV2)': (
        'results/003d_dense_eval/visualization/OWH-Dense-test', '{id}_003d_dense_eval.png'),
}

# radial bands as fraction of Nyquist (smaller-dim)
BANDS = [(0.00, 0.02), (0.02, 0.06), (0.06, 0.15), (0.15, 0.50), (0.50, 10.0)]
BAND_NAMES = ['DC~0.02', '0.02~0.06', '0.06~0.15', '0.15~0.50', '>0.50']

N_RADIAL_BINS = 64  # for the continuous curve


def radial_stats(res):
    """res: HxWx3 float residual. Returns (band fractions, radial curve)."""
    h, w = res.shape[:2]
    power = np.zeros((h, w), dtype=np.float64)
    for c in range(3):
        spec = np.fft.fftshift(np.fft.fft2(res[:, :, c]))
        power += np.abs(spec)**2
    nyq = min(h, w) / 2.0
    v, u = np.indices((h, w))
    r = np.sqrt((u - w / 2.0)**2 + (v - h / 2.0)**2) / nyq  # 1.0 ~ Nyquist

    tot = power.sum()
    npix = h * w
    fracs = [power[(r >= lo) & (r < hi)].sum() / tot for lo, hi in BANDS]
    abs_e = [power[(r >= lo) & (r < hi)].sum() / npix for lo, hi in BANDS]

    edges = np.linspace(0, 1.0, N_RADIAL_BINS + 1)
    curve = np.zeros(N_RADIAL_BINS, dtype=np.float64)
    for i in range(N_RADIAL_BINS):
        curve[i] = power[(r >= edges[i]) & (r < edges[i + 1])].sum()
    curve /= tot
    return np.array(fracs), curve, np.array(abs_e)


def main():
    ids = sorted(f.split('.')[0] for f in os.listdir(GT_DIR) if f.endswith('.jpg'))
    print(f'{len(ids)} GT images')

    results = {}
    for name, (root, tmpl) in MODELS.items():
        fracs_all, curves_all, abs_all, psnrs = [], [], [], []
        used = 0
        for img_id in ids:
            out_path = os.path.join(root, tmpl.format(id=img_id))
            if not os.path.exists(out_path):
                continue
            gt = cv2.imread(os.path.join(GT_DIR, f'{img_id}.jpg'))[:, :, ::-1]
            out = cv2.imread(out_path)[:, :, ::-1]
            if gt is None or out is None or gt.shape != out.shape:
                continue
            res = out.astype(np.float64) - gt.astype(np.float64)
            fracs, curve, abs_e = radial_stats(res)
            fracs_all.append(fracs)
            curves_all.append(curve)
            abs_all.append(abs_e)
            mse = (res**2).mean()
            psnrs.append(10 * np.log10(255.0**2 / max(mse, 1e-8)))
            used += 1
        results[name] = (np.mean(fracs_all, 0), np.mean(curves_all, 0),
                         np.mean(abs_all, 0), np.mean(psnrs), used)
        print(f'{name}: {used} imgs, mean PSNR {np.mean(psnrs):.4f}')

    # band fraction table
    print('\nresidual energy share by band (fraction of Nyquist):')
    header = 'model'.ljust(22) + ''.join(b.rjust(12) for b in BAND_NAMES)
    print(header)
    for name, (fracs, _, _, _, _) in results.items():
        print(name.ljust(22) + ''.join(f'{f * 100:11.2f}%' for f in fracs))

    print('\nabsolute residual energy per band (mean power per pixel, 255 scale):')
    print(header)
    for name, (_, _, abs_e, _, _) in results.items():
        print(name.ljust(22) + ''.join(f'{e:12.2f}' for e in abs_e))

    # radial curves plot
    edges = np.linspace(0, 1.0, N_RADIAL_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (_, curve, _, psnr, _) in results.items():
        ax.plot(centers, curve * 100, label=f'{name} (PSNR {psnr:.2f})')
    for lo, hi in BANDS[:-1]:
        ax.axvline(hi, color='gray', ls=':', lw=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('radius / Nyquist (smaller dim)')
    ax.set_ylabel('residual energy share per bin (%)')
    ax.set_title('OWH Dense-test residual radial spectrum')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'residual_spectrum_dense.png'), dpi=150)
    print(f'\nfigure saved to {OUT_DIR}/residual_spectrum_dense.png')

    # save csv of band fractions
    import csv
    with open(os.path.join(OUT_DIR, 'residual_spectrum_dense.csv'), 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['model', 'psnr'] + [f'share_{b}' for b in BAND_NAMES] + [f'abs_{b}' for b in BAND_NAMES])
        for name, (fracs, _, abs_e, psnr, _) in results.items():
            wcsv.writerow([name, f'{psnr:.4f}'] + [f'{x:.6f}' for x in fracs] + [f'{x:.4f}' for x in abs_e])


if __name__ == '__main__':
    main()
