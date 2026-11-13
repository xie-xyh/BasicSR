# DEConv branch utilization + frequency response analysis.
# For every DEConv in a DEA-Net-family checkpoint, measures:
#   (a) per-branch effective-kernel energy share (CD/HD/VD/AD/vanilla) per layer
#   (b) merged-kernel 2D frequency response, radially averaged, per level
#   (c) HD/VD/AD directional (horizontal vs vertical) energy
# Usage: python research/004_lowfreq/tools/analyze_deconv.py

import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from basicsr.archs.deanet_arch import DEConv
from basicsr.archs import build_network

OUT_DIR = 'research/004_lowfreq'

MODELS = {
    '001 (L1)': ('DEANet', {},
                 'experiments/001_DEANet_OWH-Mixed-L1_30k_B40G2/models/net_g_30000.pth'),
    '003c (L1+FFT+Grad)': ('DEANet', {},
                           'experiments/003c_DEANet_OWH-Mixed512_L1FFTGrad_60k_B20G1/models/net_g_latest.pth'),
    '004a (V3-LFM)': ('DEANetV3', {'global_block': 'lfm'},
                      'experiments/004a_DEANetV3-LFM_OWH-Mixed512_L1FFTGrad_60k_B20G1/models/net_g_60000.pth'),
}

BRANCHES = ['cd', 'hd', 'vd', 'ad', 'vanilla']


def level_of(name):
    if 'level1' in name:
        return 'level1'
    if 'level2' in name:
        return 'level2'
    return 'level3'


def branch_kernels(deconv):
    """Return dict of effective 3x3 kernels per branch (torch tensors)."""
    w_cd, _ = deconv.conv1_1.get_weight()
    w_hd, _ = deconv.conv1_2.get_weight()
    w_vd, _ = deconv.conv1_3.get_weight()
    w_ad, _ = deconv.conv1_4.get_weight()
    w_vc = deconv.conv1_5.weight
    return dict(cd=w_cd, hd=w_hd, vd=w_vd, ad=w_ad, vanilla=w_vc)


def radial_response(w, pad=64):
    """w: (Cout, Cin, 3, 3). Returns radially averaged |H|^2 (normalized)."""
    wf = torch.fft.fft2(w.float(), s=(pad, pad))  # frequency response per (co,ci)
    power = (wf.abs()**2).sum(dim=(0, 1))
    power = torch.fft.fftshift(power, dim=(0, 1)).detach().numpy()
    c = pad // 2
    v, u = np.indices((pad, pad))
    r = np.sqrt((u - c)**2 + (v - c)**2).astype(np.int32)
    prof = np.zeros(r.max() + 1, dtype=np.float64)
    cnt = np.zeros(r.max() + 1, dtype=np.float64)
    np.add.at(prof, r.ravel(), power.ravel())
    np.add.at(cnt, r.ravel(), 1)
    prof = prof / np.maximum(cnt, 1)
    return prof / prof.sum()


def band_share(prof):
    """Share of response energy in DC / low / mid / high radial bands (pad=64)."""
    return {
        'DC': prof[0],
        'low(1-8)': prof[1:9].sum(),
        'mid(9-20)': prof[9:21].sum(),
        'high(21+)': prof[21:].sum(),
    }


def analyze(name, arch, arch_kw, ckpt):
    net = build_network(dict(type=arch, base_dim=32, **arch_kw))
    sd = torch.load(ckpt, map_location='cpu')
    key = 'params_ema' if 'params_ema' in sd else 'params'
    net.load_state_dict(sd[key], strict=False)

    rows = []  # (layer_name, level, branch, energy, share)
    freq_by_level = defaultdict(list)
    dir_energy = defaultdict(lambda: np.zeros(2))  # branch -> [horizontal, vertical]
    for mod_name, mod in net.named_modules():
        if not isinstance(mod, DEConv):
            continue
        lv = level_of(mod_name)
        ks = branch_kernels(mod)
        e = {b: (k**2).sum().item() for b, k in ks.items()}
        tot = sum(e.values())
        for b in BRANCHES:
            rows.append((mod_name, lv, b, e[b], e[b] / tot))
        # merged kernel frequency response
        w_merged = sum(ks.values())
        freq_by_level[lv].append(radial_response(w_merged))
        # directionality of difference branches (positions: cols 0/2 = horiz, rows 0/2 = vert)
        for b in ('hd', 'vd', 'ad'):
            k = ks[b]
            horiz = (k[:, :, :, [0, 2]]**2).sum().item()
            vert = (k[:, :, [0, 2], :]**2).sum().item()
            dir_energy[b] += [horiz, vert]

    return rows, {lv: np.mean(v, axis=0) for lv, v in freq_by_level.items()}, dir_energy


def main():
    all_rows = {}
    all_freq = {}
    all_dir = {}
    for name, (arch, kw, ckpt) in MODELS.items():
        print(f'=== {name} ({os.path.basename(ckpt)}) ===')
        rows, freq, dire = analyze(name, arch, kw, ckpt)
        all_rows[name] = rows
        all_freq[name] = freq
        all_dir[name] = dire
        # print branch share per level
        for lv in ('level1', 'level2', 'level3'):
            shares = {b: np.mean([r[4] for r in rows if r[1] == lv and r[2] == b]) for b in BRANCHES}
            print(f'  {lv}: ' + '  '.join(f'{b} {shares[b] * 100:5.1f}%' for b in BRANCHES))
        print('  directional energy (horiz vs vert): ' +
              '  '.join(f'{b} {d[0]:.3f}/{d[1]:.3f}' for b, d in dire.items()))

    # ---- figure 1: branch share per layer (all DEConvs in order), per model ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, (name, rows) in zip(axes, all_rows.items()):
        layers = sorted({r[0] for r in rows})
        layers.sort(key=lambda n: (level_of(n), 'down' not in n, n))
        bottom = np.zeros(len(layers))
        for b in BRANCHES:
            vals = np.array([next(r[4] for r in rows if r[0] == l and r[2] == b) for l in layers])
            ax.bar(range(len(layers)), vals, bottom=bottom, label=b, width=0.85)
            bottom += vals
        ax.set_ylabel('branch energy share')
        ax.set_title(name)
        ax.set_ylim(0, 1)
    axes[0].legend(ncol=5, fontsize=8)
    axes[-1].set_xticks(range(len(layers)))
    axes[-1].set_xticklabels([l.replace('down_', 'd_').replace('up_', 'u_').replace('_block', 'b')
                              for l in layers], rotation=90, fontsize=6)
    fig.suptitle('DEConv branch energy share per layer')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'deconv_branch_share.png'), dpi=150)
    plt.close(fig)

    # ---- figure 2: merged-kernel radial frequency response per level ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    r = np.arange(32)
    for ax, lv in zip(axes, ('level1', 'level2', 'level3')):
        for name, freq in all_freq.items():
            ax.plot(np.arange(32), freq[lv][:32], label=name)
        ax.set_title(f'{lv} merged-kernel response')
        ax.set_xlabel('radius (of 64-padded FFT)')
        ax.set_yscale('log')
    axes[0].set_ylabel('normalized |H|^2')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'deconv_freq_response.png'), dpi=150)
    plt.close(fig)

    # ---- csv ----
    import csv
    with open(os.path.join(OUT_DIR, 'deconv_analysis.csv'), 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['model', 'level', 'branch_share_cd', 'share_hd', 'share_vd', 'share_ad',
                       'share_vanilla', 'resp_DC', 'resp_low', 'resp_mid', 'resp_high'])
        for name in MODELS:
            rows, freq = all_rows[name], all_freq[name]
            for lv in ('level1', 'level2', 'level3'):
                shares = [np.mean([r[4] for r in rows if r[1] == lv and r[2] == b]) for b in BRANCHES]
                bs = band_share(freq[lv])
                wcsv.writerow([name, lv] + [f'{s:.4f}' for s in shares] +
                              [f'{bs[k]:.4f}' for k in ('DC', 'low(1-8)', 'mid(9-20)', 'high(21+)')])
    print(f'\nfigures + csv saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
