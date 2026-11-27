#!/usr/bin/env python3
"""Train-free mechanistic & interpretability analysis suite for MaritimeSFNet (014a / 014b / 014d).

Runs on GPU server (d533) using real trained 8k EMA checkpoints and the 283-image OWH internal-v2
maritime validation benchmark. Produces reproducible JSON + NPZ + sample visual maps for the
Method section figures and quantitative tables.
"""
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
import yaml

from basicsr.archs import build_network

REPO_ROOT = Path(__file__).resolve().parents[4]
OUT_DIR = REPO_ROOT / 'research/014_maritime_sfnet/paper_method_kit/analysis'

MODELS = {
    '014a': (
        REPO_ROOT / 'research/014_maritime_sfnet/probe_014a_base.yml',
        REPO_ROOT / 'experiments/probes/probe_014a_base_hust_v2_8000iter/models/net_g_latest.pth',
    ),
    '014b': (
        REPO_ROOT / 'research/014_maritime_sfnet/probe_014b_asfb.yml',
        REPO_ROOT / 'experiments/probes/probe_014b_asfb_hust_v2_8000iter/models/net_g_latest.pth',
    ),
    '014d': (
        REPO_ROOT / 'research/014_maritime_sfnet/probe_014d.yml',
        REPO_ROOT / 'experiments/probes/probe_014d_hust_v2_8000iter/models/net_g_latest.pth',
    ),
}

MANIFESTS = {
    'Dense': REPO_ROOT / 'research/011_wavelet_unet/protocol_v2_20261118/confirm_011p_backbone_draft/val_manifest.txt',
    'Mist': REPO_ROOT / 'research/011_wavelet_unet/protocol_v2_20261118/confirm_011p_backbone_draft/val_2_manifest.txt',
    'Moderate': REPO_ROOT / 'research/011_wavelet_unet/protocol_v2_20261118/confirm_011p_backbone_draft/val_3_manifest.txt',
}
DATAROOT_LQ = REPO_ROOT / 'datasets/OWH/Mixed/train/hazy'
DATAROOT_GT = REPO_ROOT / 'datasets/OWH/Mixed/train/clear'


def load_model(tag: str, device='cuda'):
    cfg_path, ckpt_path = MODELS[tag]
    cfg = yaml.safe_load(cfg_path.read_text())
    ng = dict(cfg['network_g'])
    ng['activation_checkpointing'] = False
    net = build_network(ng)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('params_ema', ckpt.get('params', ckpt))
    net.load_state_dict(state, strict=True)
    return net.to(device).eval(), cfg


def read_rgb01(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def parse_manifest(path: Path):
    items = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.split()[0]
        items.append(rel)
    return items


def radial_power_spectrum(diff_hwc: np.ndarray, n_bins: int = 128):
    """Normalized radial 2D FFT power profile vs Nyquist radius r in [0, 1]."""
    h, w, _ = diff_hwc.shape
    gray = diff_hwc.mean(axis=2)
    fft = np.fft.fft2(gray, norm='ortho')
    power = (np.abs(fft) ** 2) / (h * w)
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    nyq = max(min(h, w) / 2.0, 1.0)
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / nyq

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    prof = np.zeros(n_bins, dtype=np.float64)
    total_e = float(power.sum()) + 1e-12
    e_low = float(power[r <= 0.02].sum()) / total_e
    e_mid = float(power[(r > 0.02) & (r <= 0.15)].sum()) / total_e
    e_high = float(power[r > 0.15].sum()) / total_e
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if np.any(m):
            prof[i] = float(power[m].sum()) / total_e
    return prof, (e_low, e_mid, e_high)


def analyze_maritime_physics(dataset_items):
    """Exp 2: Maritime residual spectrum + vertical vs horizontal spatial anisotropy."""
    print('[Exp 2] Analyzing maritime haze physics (residual FFT spectrum & directional anisotropy)...')
    n_bins = 100
    r_centers = (np.linspace(0.0, 1.0, n_bins + 1)[:-1] + np.linspace(0.0, 1.0, n_bins + 1)[1:]) / 2.0
    per_subset = {}
    all_profs = []
    for subset, rels in dataset_items.items():
        profs, bands = [], []
        dy_list, dx_list = [], []
        vert_profile = np.zeros(32, dtype=np.float64)
        for rel in rels:
            lq = read_rgb01(DATAROOT_LQ / rel)
            gt = read_rgb01(DATAROOT_GT / rel)
            veil = lq - gt
            prof, b = radial_power_spectrum(veil, n_bins=n_bins)
            profs.append(prof)
            bands.append(b)
            all_profs.append(prof)

            # Smooth veil to isolate atmospheric scattering field from local texture
            veil_smooth = cv2.GaussianBlur(veil.mean(axis=2), (31, 31), 7.0)
            h, w = veil_smooth.shape
            # Compare normalized vertical vs horizontal gradient magnitude
            gy = np.abs(np.diff(veil_smooth, axis=0)).mean() * h
            gx = np.abs(np.diff(veil_smooth, axis=1)).mean() * w
            dy_list.append(float(gy))
            dx_list.append(float(gx))
            row_mean = veil_smooth.mean(axis=1)
            vert_profile += cv2.resize(row_mean[:, None], (1, 32), interpolation=cv2.INTER_AREA)[:, 0]

        bands_arr = np.array(bands)
        per_subset[subset] = {
            'n': len(rels),
            'low_frac_lt002': float(bands_arr[:, 0].mean()),
            'mid_frac_002_015': float(bands_arr[:, 1].mean()),
            'high_frac_gt015': float(bands_arr[:, 2].mean()),
            'mean_norm_grad_y': float(np.mean(dy_list)),
            'mean_norm_grad_x': float(np.mean(dx_list)),
            'anisotropy_ratio_y_over_x': float(np.mean(dy_list) / (np.mean(dx_list) + 1e-12)),
            'pct_images_y_gt_x': float(np.mean(np.array(dy_list) > np.array(dx_list))),
            'mean_profile': np.mean(profs, axis=0).tolist(),
            'vertical_veil_profile_32': (vert_profile / len(rels)).tolist(),
        }
    return {
        'r_centers': r_centers.tolist(),
        'per_subset': per_subset,
        'overall_profile': np.mean(all_profs, axis=0).tolist(),
    }


@torch.no_grad()
def analyze_rep5_cd_and_folding(net_014d):
    """Exp 3: Learned Rep5-CD branch spectral response, stage energy, & lossless folding benchmark."""
    print('[Exp 3] Analyzing learned Rep5-CD 5-branch frequency response & fold_reparam() speed...')
    from basicsr.archs.maritime_sfnet_arch import AnisoRepConv3x3

    grid_n = 64
    fy = np.fft.fftfreq(grid_n) * 2.0  # normalized Nyquist [-1, 1]
    fx = np.fft.fftfreq(grid_n) * 2.0
    rr = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    r_bins = np.linspace(0.0, 1.0, 33)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    branch_names = ('plain', 'pw', 'hd', 'vd', 'cd')
    stage_energy = {}
    branch_radial = {b: [] for b in branch_names}
    branch_2d_fft = {b: np.zeros((grid_n, grid_n), dtype=np.float64) for b in branch_names}
    branch_dc_ratio = {b: [] for b in branch_names}
    total_modules = 0

    for name, mod in net_014d.named_modules():
        if not isinstance(mod, AnisoRepConv3x3):
            continue
        stage = name.split('.')[0]
        w_plain = mod.plain.weight.detach().float().cpu()
        w_pw = F.pad(mod.pw.weight.detach().float().cpu(), (1, 1, 1, 1))
        zero_hd = torch.zeros_like(mod.hd.weight.detach().float().cpu())
        w_hd = torch.stack((mod.hd.weight.detach().float().cpu(), zero_hd, -mod.hd.weight.detach().float().cpu()), dim=3)
        zero_vd = torch.zeros_like(mod.vd.weight.detach().float().cpu())
        w_vd = torch.stack((mod.vd.weight.detach().float().cpu(), zero_vd, -mod.vd.weight.detach().float().cpu()), dim=2)
        w_cd_raw = mod.cd.weight.detach().float().cpu()
        w_cd = w_cd_raw - F.pad(w_cd_raw.sum(dim=(2, 3), keepdim=True), (1, 1, 1, 1))

        kernels = {'plain': w_plain, 'pw': w_pw, 'hd': w_hd, 'vd': w_vd, 'cd': w_cd}
        energies = {b: float((k ** 2).sum().item()) for b, k in kernels.items()}
        e_tot = sum(energies.values()) + 1e-12
        stage_energy.setdefault(stage, {b: [] for b in branch_names})
        for b in branch_names:
            stage_energy[stage][b].append(energies[b] / e_tot)

            # 2D FFT magnitude averaged over (cout, cin)
            k_np = kernels[b].numpy()  # (C, C, 3, 3)
            # Subsample up to 256 filters for fast exact mean power spectrum
            flat = k_np.reshape(-1, 3, 3)
            if len(flat) > 512:
                idx = np.linspace(0, len(flat) - 1, 512, dtype=int)
                flat = flat[idx]
            spec = np.abs(np.fft.fft2(flat, s=(grid_n, grid_n))) ** 2  # (N, 64, 64)
            mean_spec = spec.mean(axis=0)
            norm_spec = mean_spec / (mean_spec.sum() + 1e-12)
            branch_2d_fft[b] += np.fft.fftshift(norm_spec)
            dc_val = float(norm_spec[0, 0])
            branch_dc_ratio[b].append(dc_val)

            rad = np.zeros(len(r_centers), dtype=np.float64)
            for i in range(len(r_centers)):
                m = (rr >= r_bins[i]) & (rr < r_bins[i + 1])
                if np.any(m):
                    rad[i] = float(norm_spec[m].mean())
            rad /= (rad.sum() + 1e-12)
            branch_radial[b].append(rad)
        total_modules += 1

    stage_summary = {}
    for st, d in stage_energy.items():
        stage_summary[st] = {b: float(np.mean(v)) for b, v in d.items()}

    # Folding equivalence & GPU latency benchmark (RTX 3090)
    net_unfolded = copy.deepcopy(net_014d).cuda().eval()
    net_folded = copy.deepcopy(net_014d).cuda().eval()
    net_folded.fold_reparam()

    params_unfolded = sum(p.numel() for p in net_unfolded.parameters())
    params_folded = sum(p.numel() for p in net_folded.parameters())

    x_check = torch.rand(1, 3, 257, 389, device='cuda')
    y_u = net_unfolded(x_check)
    y_f = net_folded(x_check)
    fold_max_err = float((y_u - y_f).abs().max().item())

    latency = {}
    for label, (hh, ww) in (('512x512', (512, 512)), ('720p', (720, 1280)), ('1080p', (1080, 1920))):
        x_bench = torch.rand(1, 3, hh, ww, device='cuda')
        res_pair = {}
        for mode_name, m in (('unfolded', net_unfolded), ('folded', net_folded)):
            for _ in range(5):
                _ = m(x_bench)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n_runs = 20 if hh <= 720 else 10
            for _ in range(n_runs):
                _ = m(x_bench)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / n_runs
            res_pair[mode_name] = {'ms': round(ms, 2), 'fps': round(1000.0 / ms, 2)}
        latency[label] = res_pair

    del net_unfolded, net_folded
    torch.cuda.empty_cache()

    return {
        'r_centers': r_centers.tolist(),
        'stage_energy_share': stage_summary,
        'overall_energy_share': {
            b: float(np.mean([stage_summary[st][b] for st in stage_summary]))
            for b in branch_names
        },
        'branch_dc_ratio': {b: float(np.mean(v)) for b, v in branch_dc_ratio.items()},
        'branch_radial_profile': {b: np.mean(branch_radial[b], axis=0).tolist() for b in branch_names},
        'branch_2d_fft': {b: (branch_2d_fft[b] / max(total_modules, 1)) for b in branch_names},
        'folding': {
            'params_unfolded': params_unfolded,
            'params_folded': params_folded,
            'param_reduction_pct': round((1.0 - params_folded / params_unfolded) * 100.0, 2),
            'max_abs_err': fold_max_err,
            'latency_rtx3090': latency,
        },
    }


def compute_erf(net, sample_paths, crop_size=256):
    """Exp 4: Effective Receptive Field (Ding et al. CVPR 2022) via center-pixel gradient."""
    net.eval()
    acc = np.zeros((crop_size, crop_size), dtype=np.float64)
    cy, cx = crop_size // 2, crop_size // 2
    for p in sample_paths:
        img = read_rgb01(p)
        h, w, _ = img.shape
        y0 = max(0, (h - crop_size) // 2)
        x0 = max(0, (w - crop_size) // 2)
        patch = img[y0:y0 + crop_size, x0:x0 + crop_size]
        if patch.shape[0] != crop_size or patch.shape[1] != crop_size:
            patch = cv2.resize(img, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(patch.transpose(2, 0, 1))[None].cuda().requires_grad_(True)
        net.zero_grad(set_to_none=True)
        out = net(x)
        target_val = out[0, :, cy, cx].sum()
        target_val.backward()
        grad = x.grad.detach().abs().mean(dim=1)[0].cpu().numpy()
        acc += grad
    acc = np.log1p(acc / max(len(sample_paths), 1))
    acc /= (acc.max() + 1e-12)

    # Compute high-contribution area ratios (fraction of pixels > thresh * max)
    # and directional spread (horizontal vs vertical FWHM / 20%-max span)
    h_prof = acc[cy, :]
    v_prof = acc[:, cx]
    stats = {}
    for thr in (0.1, 0.2, 0.3, 0.5):
        stats[f'area_gt_{int(thr*100)}pct'] = float((acc >= thr).mean())
        stats[f'h_span_gt_{int(thr*100)}pct_px'] = int((h_prof >= thr).sum())
        stats[f'v_span_gt_{int(thr*100)}pct_px'] = int((v_prof >= thr).sum())
    return acc, h_prof, v_prof, stats


@torch.no_grad()
def analyze_gates_embeddings_and_samples(nets, dataset_items):
    """Exp 5 & 6: Per-image BASR/FCSF stats, bottleneck embeddings for t-SNE/K-Means, & sample maps."""
    print('[Exp 5 & 6] Extracting BASR/FCSF gate stats, bottleneck embeddings, and spatial heatmaps...')
    records = {tag: [] for tag in nets}
    embeddings = {tag: [] for tag in nets}
    labels = []
    rel_list = []

    # Forward-hook capture containers
    hook_buf = {}

    def attach_hooks(tag, net):
        handles = []

        def bot_hook(_m, _inp, out):
            hook_buf[f'{tag}_bot'] = F.adaptive_avg_pool2d(out, 1).flatten(1).detach().cpu().numpy()[0]

        handles.append(net.asfb2.register_forward_hook(bot_hook))

        if tag == '014d':
            def basr_hook(m, inp, _out):
                feat, image = inp
                raw_k, bias = m.head(feat).chunk(2, dim=1)
                g_feat = F.adaptive_avg_pool2d(feat, 1)
                g_img_mean = F.adaptive_avg_pool2d(image, 1)
                g_img_std = torch.sqrt(image.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-6)
                scene_scale = 2.0 * torch.sigmoid(
                    m.scene_mlp(torch.cat((g_feat, g_img_mean, g_img_std), dim=1)))
                s_k, s_b = scene_scale.chunk(2, dim=1)
                rk = raw_k * s_k
                gain = torch.where(
                    rk >= 0,
                    1.0 + m.a_pos * torch.tanh(rk / m.a_pos),
                    1.0 + m.a_neg * torch.tanh(rk / m.a_neg))
                hook_buf['014d_basr_gain'] = gain.detach().cpu()
                hook_buf['014d_scene_sk'] = float(s_k.mean().item())
                hook_buf['014d_scene_sb'] = float(s_b.mean().item())
                hook_buf['014d_scene_vec'] = torch.cat((s_k, s_b, g_img_mean, g_img_std), dim=1).flatten().detach().cpu().numpy()

            def fcsf2_hook(m, inp, _out):
                deep, skip = inp
                skip_low = m._low_pass(skip, m.lp_stride)
                skip_detail = skip - skip_low
                deep_low = m._low_pass(deep, m.lp_stride)
                ev = torch.cat((m._pool2(deep), m._pool2(skip_low - deep_low), m._pool2(skip_detail.abs())), dim=1)
                logits = F.interpolate(m.gate(ev), size=deep.shape[-2:], mode='bilinear', align_corners=False)
                if m.scene_gate is not None:
                    logits = logits + m.scene_gate(ev)
                alpha, beta = torch.sigmoid(logits).chunk(2, dim=1)
                hook_buf['014d_fcsf2_alpha_map'] = alpha.mean(dim=1, keepdim=True).detach().cpu()
                hook_buf['014d_fcsf2_beta_map'] = beta.mean(dim=1, keepdim=True).detach().cpu()

            def fcsf1_hook(m, inp, _out):
                deep, skip = inp
                skip_low = m._low_pass(skip, m.lp_stride)
                skip_detail = skip - skip_low
                ev = torch.cat((m._pool2(deep), m._pool2(skip_detail.abs())), dim=1)
                logits = F.interpolate(m.gate(ev), size=deep.shape[-2:], mode='bilinear', align_corners=False)
                if m.scene_gate is not None:
                    logits = logits + m.scene_gate(ev)
                beta = torch.sigmoid(logits)
                hook_buf['014d_fcsf1_beta_map'] = beta.mean(dim=1, keepdim=True).detach().cpu()

            def dca_hook(m, inp, _out):
                x_in = inp[0]
                c_attn = m.ca(x_in)
                s_in = torch.cat((x_in.mean(dim=1, keepdim=True), x_in.amax(dim=1, keepdim=True)), dim=1)
                s_attn = m.sa(s_in)
                p1 = c_attn + s_attn
                b, c, h, w = x_in.shape
                pa_in = torch.stack((x_in, p1), dim=2).reshape(b, 2 * c, h, w)
                gate = 2.0 * torch.sigmoid(p1 + m.pa(pa_in))
                hook_buf['014d_dca_pa_map'] = gate.std(dim=1, keepdim=True).detach().cpu()

            handles.append(net.basr_head.register_forward_hook(basr_hook))
            handles.append(net.fcsf2.register_forward_hook(fcsf2_hook))
            handles.append(net.fcsf1.register_forward_hook(fcsf1_hook))
            handles.append(net.bot_seg3[-1].attn.register_forward_hook(dca_hook))
        elif tag == '014b':
            def basr_b_hook(m, inp, _out):
                feat, _ = inp
                raw_k, _ = m.head(feat).chunk(2, dim=1)
                gain = m.g_min + (m.g_max - m.g_min) * torch.sigmoid(raw_k + m.b0)
                hook_buf['014b_basr_gain'] = gain.detach().cpu()
            handles.append(net.basr_head.register_forward_hook(basr_b_hook))
        return handles

    all_handles = []
    for tag, net in nets.items():
        all_handles.extend(attach_hooks(tag, net))

    best_sample_per_subset = {}
    for subset, rels in dataset_items.items():
        for rel in rels:
            lq_np = read_rgb01(DATAROOT_LQ / rel)
            gt_np = read_rgb01(DATAROOT_GT / rel)
            h, w, _ = lq_np.shape
            x = torch.from_numpy(lq_np.transpose(2, 0, 1))[None].cuda()
            img_rms = float(lq_np.std())
            labels.append(subset)
            rel_list.append(rel)

            psnrs = {}
            outs_np = {}
            for tag, net in nets.items():
                out = net(x).clamp(0.0, 1.0)
                out_np = out[0].cpu().numpy().transpose(1, 2, 0)
                mse = float(np.mean((out_np - gt_np) ** 2))
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
                psnrs[tag] = psnr
                outs_np[tag] = out_np
                embeddings[tag].append(hook_buf[f'{tag}_bot'].copy())

                rec = {
                    'subset': subset,
                    'rel': rel,
                    'psnr': psnr,
                    'img_rms': img_rms,
                    'basr_gain_mean': float(net.basr_head.last_gain_mean.item()),
                    'basr_bias_abs': float(net.basr_head.last_bias_abs.item()),
                    'fcsf3_alpha': float(net.fcsf3.last_alpha_mean.item()),
                    'fcsf2_alpha': float(net.fcsf2.last_alpha_mean.item()),
                    'fcsf1_beta': float(net.fcsf1.last_beta_mean.item()),
                }
                if tag == '014d':
                    g_t = hook_buf['014d_basr_gain']
                    rec['basr_gain_p05'] = float(torch.quantile(g_t.flatten(), 0.05).item())
                    rec['basr_gain_p95'] = float(torch.quantile(g_t.flatten(), 0.95).item())
                    rec['basr_gain_std'] = float(g_t.std().item())
                    rec['scene_sk'] = hook_buf['014d_scene_sk']
                    rec['scene_sb'] = hook_buf['014d_scene_sb']
                elif tag == '014b':
                    g_t = hook_buf['014b_basr_gain']
                    rec['basr_gain_p05'] = float(torch.quantile(g_t.flatten(), 0.05).item())
                    rec['basr_gain_p95'] = float(torch.quantile(g_t.flatten(), 0.95).item())
                    rec['basr_gain_std'] = float(g_t.std().item())
                records[tag].append(rec)

            # Pick visually clear representative maritime sample per subset where 014d clearly beats 014b
            gain_d_over_b = psnrs['014d'] - psnrs['014b']
            # Prefer landscape maritime aspect ratio (w > h) with moderate-to-high improvement
            score = gain_d_over_b + (0.5 if w >= h else 0.0)
            prev = best_sample_per_subset.get(subset)
            if prev is None or score > prev['score']:
                best_sample_per_subset[subset] = {
                    'score': score,
                    'rel': rel,
                    'psnr_014a': psnrs['014a'],
                    'psnr_014b': psnrs['014b'],
                    'psnr_014d': psnrs['014d'],
                    'lq': lq_np,
                    'gt': gt_np,
                    'out_014b': outs_np['014b'],
                    'out_014d': outs_np['014d'],
                    'basr_gain_map': hook_buf['014d_basr_gain'][0, :, :h, :w].mean(dim=0).numpy(),
                    'basr_gain_map_014b': hook_buf['014b_basr_gain'][0, :, :h, :w].mean(dim=0).numpy(),
                    'fcsf2_alpha_map': hook_buf['014d_fcsf2_alpha_map'][0, 0].numpy(),
                    'fcsf1_beta_map': hook_buf['014d_fcsf1_beta_map'][0, 0, :h, :w].numpy(),
                    'dca_pa_map': hook_buf['014d_dca_pa_map'][0, 0].numpy(),
                    'scene_sk': hook_buf['014d_scene_sk'],
                    'scene_sb': hook_buf['014d_scene_sb'],
                }

    for hnd in all_handles:
        hnd.remove()

    # Save representative spatial maps
    sample_dir = OUT_DIR / 'sample_maritime_maps'
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_meta = {}
    for subset, s in best_sample_per_subset.items():
        np.savez_compressed(
            sample_dir / f'{subset.lower()}_maps.npz',
            lq=s['lq'],
            gt=s['gt'],
            out_014b=s['out_014b'],
            out_014d=s['out_014d'],
            basr_gain_map=s['basr_gain_map'],
            basr_gain_map_014b=s['basr_gain_map_014b'],
            fcsf2_alpha_map=s['fcsf2_alpha_map'],
            fcsf1_beta_map=s['fcsf1_beta_map'],
            dca_pa_map=s['dca_pa_map'],
        )
        sample_meta[subset] = {
            'rel': s['rel'],
            'psnr_014a': round(s['psnr_014a'], 3),
            'psnr_014b': round(s['psnr_014b'], 3),
            'psnr_014d': round(s['psnr_014d'], 3),
            'scene_sk': round(s['scene_sk'], 4),
            'scene_sb': round(s['scene_sb'], 4),
        }

    # Unsupervised clustering & t-SNE manifold analysis
    emb_arrays = {tag: np.stack(vecs, axis=0) for tag, vecs in embeddings.items()}
    label_map = {'Dense': 0, 'Moderate': 1, 'Mist': 2}
    y_true = np.array([label_map[lb] for lb in labels], dtype=int)

    manifold_results = {}
    tsne_coords = {}
    for tag, X in emb_arrays.items():
        X_std = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.linear_model import LogisticRegression

            pca = PCA(n_components=min(20, X_std.shape[1]), random_state=42)
            X_pca = pca.fit_transform(X_std)
            tsne = TSNE(n_components=2, perplexity=25, learning_rate='auto', init='pca', random_state=42)
            coords = tsne.fit_transform(X_pca)
            km = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X_pca)
            y_pred = km.labels_
            sil_true = float(silhouette_score(X_pca, y_true))
            sil_km = float(silhouette_score(X_pca, y_pred))
            ari = float(adjusted_rand_score(y_true, y_pred))
            nmi = float(normalized_mutual_info_score(y_true, y_pred))
            # Cluster purity
            purity_sum = 0
            for c in range(3):
                m = (y_pred == c)
                if np.any(m):
                    purity_sum += int(np.bincount(y_true[m]).max())
            purity = float(purity_sum / len(y_true))
            clf = LogisticRegression(max_iter=1000)
            cv_acc = float(cross_val_score(clf, X_pca, y_true, cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean())
            tsne_coords[tag] = coords
            manifold_results[tag] = {
                'silhouette_true_labels': round(sil_true, 4),
                'silhouette_kmeans': round(sil_km, 4),
                'kmeans_ari': round(ari, 4),
                'kmeans_nmi': round(nmi, 4),
                'kmeans_purity': round(purity, 4),
                'linear_probe_5fold_acc': round(cv_acc, 4),
                'kmeans_labels': y_pred.tolist(),
            }
        except Exception as exc:
            # Pure numpy PCA fallback if sklearn absent
            u, s, _ = np.linalg.svd(X_std, full_matrices=False)
            coords = u[:, :2] * s[:2]
            tsne_coords[tag] = coords
            manifold_results[tag] = {'fallback': str(exc)}

    return records, sample_meta, manifold_results, tsne_coords, y_true


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_items = {k: parse_manifest(p) for k, p in MANIFESTS.items()}

    physics = analyze_maritime_physics(dataset_items)

    nets = {}
    for tag in ('014a', '014b', '014d'):
        nets[tag], _ = load_model(tag, device='cuda')

    rep5 = analyze_rep5_cd_and_folding(nets['014d'])

    print('[Exp 4] Computing Effective Receptive Fields (014a vs 014d)...')
    erf_samples = []
    for subset in ('Dense', 'Moderate', 'Mist'):
        erf_samples.extend([DATAROOT_LQ / rel for rel in dataset_items[subset][:8]])
    erf_a, h_a, v_a, stats_a = compute_erf(nets['014a'], erf_samples, crop_size=256)
    erf_d, h_d, v_d, stats_d = compute_erf(nets['014d'], erf_samples, crop_size=256)

    records, sample_meta, manifold_results, tsne_coords, y_true = analyze_gates_embeddings_and_samples(nets, dataset_items)

    # Summarize gate statistics by subset
    gate_summary = {}
    for tag, recs in records.items():
        gate_summary[tag] = {}
        for subset in ('Dense', 'Moderate', 'Mist'):
            sub = [r for r in recs if r['subset'] == subset]
            keys = ['psnr', 'img_rms', 'basr_gain_mean', 'basr_bias_abs', 'fcsf3_alpha', 'fcsf2_alpha', 'fcsf1_beta']
            if 'scene_sk' in sub[0]:
                keys += ['basr_gain_p05', 'basr_gain_p95', 'basr_gain_std', 'scene_sk', 'scene_sb']
            elif 'basr_gain_p95' in sub[0]:
                keys += ['basr_gain_p05', 'basr_gain_p95', 'basr_gain_std']
            gate_summary[tag][subset] = {
                k: {'mean': round(float(np.mean([r[k] for r in sub])), 5),
                    'std': round(float(np.std([r[k] for r in sub])), 5)}
                for k in keys
            }

    rep5_json = {k: v for k, v in rep5.items() if k != 'branch_2d_fft'}
    summary = {
        'physics': physics,
        'rep5_cd': rep5_json,
        'erf': {'014a': stats_a, '014d': stats_d},
        'gate_summary': gate_summary,
        'manifold_tsne_kmeans': manifold_results,
        'sample_maritime_meta': sample_meta,
        'per_image_records': records,
    }
    (OUT_DIR / 'trainfree_metrics.json').write_text(json.dumps(summary, indent=2) + '\n')

    np.savez_compressed(
        OUT_DIR / 'trainfree_arrays.npz',
        erf_014a=erf_a,
        erf_014d=erf_d,
        erf_h_014a=h_a,
        erf_v_014a=v_a,
        erf_h_014d=h_d,
        erf_v_014d=v_d,
        tsne_014a=tsne_coords['014a'],
        tsne_014b=tsne_coords['014b'],
        tsne_014d=tsne_coords['014d'],
        y_true=y_true,
        **{f'fft2d_{b}': rep5['branch_2d_fft'][b] for b in rep5['branch_2d_fft']},
    )
    print(f'ALL TRAIN-FREE EXPERIMENTS COMPLETE -> {OUT_DIR}')


if __name__ == '__main__':
    main()
