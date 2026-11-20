import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric
from basicsr.utils import tensor2img


def load_network(config, checkpoint, device):
    net = build_network(config['network_g'])
    net.load_state_dict(torch.load(checkpoint, map_location='cpu')['params_ema'])
    return net.to(device).eval()


def suppress_skip(args):
    features, = args
    decoded, skip = features.chunk(2, dim=1)
    return (torch.cat((decoded, torch.zeros_like(skip)), dim=1),)


def predict_float(net, image, top=0, left=0):
    height, width = image.shape[-2:]
    pad = (left, -(width + left) % 4, top, -(height + top) % 4)
    padded = torch.nn.functional.pad(image[None], pad, mode='reflect')
    return net(padded)[..., top:top + height, left:left + width]


def tiled_float(net, image, tile=512, overlap=128):
    height, width = image.shape[-2:]

    def starts(length):
        last = max(0, length - tile)
        return sorted(set(range(0, last + 1, tile - overlap)) | {last})

    merged = torch.zeros_like(image)
    weights = torch.zeros_like(image[:1])
    for top in starts(height):
        for left in starts(width):
            bottom, right = min(top + tile, height), min(left + tile, width)
            piece = image[:, top:bottom, left:right]
            prediction = predict_float(net, piece)[0]
            ramp_y = torch.ones(bottom - top, device=image.device, dtype=image.dtype)
            ramp_x = torch.ones(right - left, device=image.device, dtype=image.dtype)
            for ramp, initial, final, size in ((ramp_y, top, bottom, height), (ramp_x, left, right, width)):
                span = min(overlap, len(ramp))
                fade = torch.linspace(0.01, 1, span, device=image.device, dtype=image.dtype)
                if initial > 0:
                    ramp[:span] *= fade
                if final < size:
                    ramp[-span:] *= fade.flip(0)
            window = ramp_y[:, None] * ramp_x[None, :]
            merged[:, top:bottom, left:right] += prediction * window
            weights[:, top:bottom, left:right] += window
    return (merged / weights)[None]


def predict(net, image, phase_average=False, smooth_scale=1):
    result = predict_float(net, image)
    if phase_average:
        for top, left in ((0, 1), (1, 0), (1, 1)):
            result += predict_float(net, image, top, left)
        result = result / 4
    if smooth_scale != 1:
        correction = (result[0] - image).permute(1, 2, 0).cpu().numpy()
        smooth = cv2.GaussianBlur(correction, (0, 0), 8)
        adjustment = torch.from_numpy(smooth).permute(2, 0, 1).to(result.device)
        result = result + (smooth_scale - 1) * adjustment
    return tensor2img(result)


def band_mse(img, gt, mid_profile=False):
    residual = (img.astype(np.float64) - gt) / 255
    height, width = residual.shape[:2]
    radius = np.hypot(2 * np.fft.fftfreq(height)[:, None], 2 * np.fft.fftfreq(width)[None, :])
    spectrum = np.fft.fft2(residual, axes=(0, 1), norm='ortho')
    power = np.abs(spectrum) ** 2
    low = power[radius < 0.02].sum() / residual.size
    high = power[radius >= 0.15].sum() / residual.size
    result = dict(low_mse=low, mid_mse=power.sum() / residual.size - low - high, high_mse=high)
    if mid_profile:
        band = (radius >= 0.02) & (radius < 0.15)
        filtered = np.fft.ifft2(spectrum * band[..., None], axes=(0, 1), norm='ortho').real
        row_energy = np.square(filtered).sum(axis=(1, 2))
        for name, stripe in zip(('top', 'middle', 'bottom'), np.array_split(row_energy, 3)):
            result[f'{name}_mid_contribution'] = stripe.sum() / residual.size
    return result


def smooth_difference(first, second):
    return cv2.GaussianBlur((first.astype(np.float32) - second) / 255, (0, 0), 8)


def correction_errors(img, gt, required):
    remaining = smooth_difference(gt, img)
    active = np.abs(required) >= 2 / 255
    squared = np.square(remaining)
    opposite = required * remaining < 0
    return dict(overshoot_mse=float(np.where(active & opposite, squared, 0).mean()),
                undershoot_mse=float(np.where(active & ~opposite, squared, 0).mean()),
                near_clear_mse=float(np.where(~active, squared, 0).mean()))


def spatial_errors(img, gt):
    residual = (img.astype(np.float32) - gt) / 255
    smooth = cv2.GaussianBlur(residual, (0, 0), 8)
    stats = {}
    for name, stripes in (('rgb', np.array_split(residual, 3, axis=0)),
                          ('smooth', np.array_split(smooth, 3, axis=0))):
        for position, stripe in zip(('top', 'middle', 'bottom'), stripes):
            stats[f'{position}_{name}_mse'] = float(np.square(stripe).mean())
    return stats


def grid_boundary_error(img, gt):
    residual = (img.astype(np.float64) - gt) / 255
    row_jumps = np.diff(residual, axis=0) ** 2
    col_jumps = np.diff(residual, axis=1) ** 2
    on_grid = (row_jumps[1::2].mean() + col_jumps[:, 1::2].mean()) / 2
    inside = (row_jumps[::2].mean() + col_jumps[:, ::2].mean()) / 2
    return on_grid - inside


def shift_error(net, image, prediction, phase_average=False, smooth_scale=1):
    border = 16
    shifted = predict(net, image[..., 1:, 1:], phase_average, smooth_scale)
    shifted = shifted[border:-border, border:-border].astype(np.float64)
    aligned = prediction[border + 1:-border, border + 1:-border]
    return np.mean(((shifted - aligned) / 255) ** 2)


def summarize(rows):
    rng = np.random.default_rng(0)
    summary = {}
    for split in sorted({row['dataset'] for row in rows}):
        pairs = [row for row in rows if row['dataset'] == split]
        summary[split] = dict(n=len(pairs))
        keys = [name.removeprefix('control_') for name in pairs[0] if name.startswith('control_')]
        for key in keys:
            baseline = np.array([row[f'control_{key}'] for row in pairs])
            candidate = np.array([row[f'candidate_{key}'] for row in pairs])
            delta = candidate - baseline
            if not np.isfinite(delta).all():
                raise ValueError(f'{split}: {key} has a non-finite difference')
            unique = sorted({row['gt_sha256'] for row in pairs})
            indices = {key: index for index, key in enumerate(unique)}
            codes = [indices[row['gt_sha256']] for row in pairs]
            group_total = np.bincount(codes, weights=delta)
            group_size = np.bincount(codes)
            draws = rng.integers(len(unique), size=(3000, len(unique)))
            means = group_total[draws].sum(axis=1) / group_size[draws].sum(axis=1)
            summary[split][key] = dict(control=float(baseline.mean()), candidate=float(candidate.mean()),
                                       delta=float(delta.mean()), median_delta=float(np.median(delta)),
                                       ci95=np.quantile(means, [0.025, 0.975]).tolist(), groups=len(unique))
    return summary


@torch.inference_mode()
def evaluate(args):
    control_cfg = yaml.safe_load(args.control_config.read_text())
    candidate_cfg = yaml.safe_load(args.candidate_config.read_text())
    baseline = load_network(control_cfg, args.control_weights, args.device)
    candidate = load_network(candidate_cfg, args.candidate_weights, args.device)
    if args.candidate_drop_skip:
        getattr(candidate, args.candidate_drop_skip).register_forward_pre_hook(
            lambda module, inputs: suppress_skip(inputs))
    metric_cfg = {
        name: control_cfg['val']['metrics'].get(name, dict(
            type=f'calculate_{name}', crop_border=0, test_y_channel=False))
        for name in ('psnr', 'ssim')
    }
    rows = []
    for phase, dataset_opt in control_cfg['datasets'].items():
        if not phase.startswith('val'):
            continue
        dataset = build_dataset(dict(dataset_opt, phase='val', scale=1))
        count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        for index in range(count):
            item = dataset[index]
            image = item['lq'].to(args.device)
            gt = tensor2img(item['gt'])
            gt_path = Path(item['gt_path'])
            row = dict(dataset=dataset_opt['name'], image=gt_path.name,
                       gt_sha256=hashlib.sha256(gt_path.read_bytes()).hexdigest())
            if args.correction_check:
                required = smooth_difference(gt, tensor2img(item['lq']))
                row['input_smooth_mse'] = float(np.square(required).mean())
            for label, net in (('control', baseline), ('candidate', candidate)):
                phase_average = label == 'candidate' and args.candidate_phase_average
                smooth_scale = args.candidate_smooth_scale if label == 'candidate' else 1
                prediction = predict(net, image, phase_average, smooth_scale)
                if label == 'candidate' and args.candidate_tiles:
                    prediction = tensor2img(tiled_float(net, image))
                measured = {name: calculate_metric(dict(img=prediction, img2=gt), opt)
                            for name, opt in metric_cfg.items()}
                measured['grid_boundary_error'] = grid_boundary_error(prediction, gt)
                if args.phase_check:
                    measured['shift_error'] = shift_error(net, image, prediction, phase_average, smooth_scale)
                if args.correction_check:
                    measured.update(correction_errors(prediction, gt, required))
                if args.spatial_check:
                    measured.update(spatial_errors(prediction, gt))
                measured.update(band_mse(prediction, gt, args.mid_profile))
                row.update({f'{label}_{name}': value for name, value in measured.items()})
            rows.append(row)
        print(f'{dataset_opt["name"]}: {count} pairs', flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'summary.json').write_text(json.dumps(summarize(rows), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for argument in ('control_config', 'control_weights', 'candidate_config', 'candidate_weights', 'output'):
        parser.add_argument(argument, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--phase-check', action='store_true')
    parser.add_argument('--correction-check', action='store_true')
    parser.add_argument('--spatial-check', action='store_true')
    parser.add_argument('--mid-profile', action='store_true')
    parser.add_argument('--candidate-phase-average', action='store_true')
    parser.add_argument('--candidate-tiles', action='store_true')
    parser.add_argument('--candidate-drop-skip', choices=('dec1', 'dec2', 'dec3'))
    parser.add_argument('--candidate-smooth-scale', type=float, default=1)
    cv2.setNumThreads(1)
    args = parser.parse_args()
    if args.candidate_tiles and (args.phase_check or args.candidate_phase_average or args.candidate_smooth_scale != 1):
        parser.error('--candidate-tiles must be evaluated on its own.')
    evaluate(args)
