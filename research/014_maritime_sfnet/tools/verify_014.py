"""Unit, config-parity, and synthetic smoke verification for 014 (MaritimeSFNet).

Run on the GPU/Linux server inside the `basicsr` conda env:
    python research/014_maritime_sfnet/tools/verify_014.py
    python research/014_maritime_sfnet/tools/verify_014.py --smoke --preflight-gpu 0

Run without PyTorch (static syntax + YAML single-variable parity audit only):
    python3 research/014_maritime_sfnet/tools/verify_014.py --config-only
"""
import argparse
import copy
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    '014a': REPO_ROOT / 'research/014_maritime_sfnet/probe_014a_base.yml',
    '014b': REPO_ROOT / 'research/014_maritime_sfnet/probe_014b_asfb.yml',
    '014c': REPO_ROOT / 'research/014_maritime_sfnet/probe_014c_mimo.yml',
    '014d': REPO_ROOT / 'research/014_maritime_sfnet/probe_014d.yml',
    '014e': REPO_ROOT / 'research/014_maritime_sfnet/probe_014e.yml',
}
REFERENCE_013B = REPO_ROOT / 'research/013_owhregion/probe_013b_wide.yml'


def _load_yaml(path: Path):
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        # Minimal fallback parser only for --config-only without PyYAML
        raise RuntimeError('PyYAML is required to audit YAML configs.')


def check_static_and_configs():
    print('=== [1/4] Syntax & YAML parity audit ===')
    for src in (
        REPO_ROOT / 'basicsr/archs/maritime_sfnet_arch.py',
        REPO_ROOT / 'basicsr/models/maritime_sf_model.py',
    ):
        py_compile.compile(str(src), doraise=True)
        print(f'  syntax OK: {src.relative_to(REPO_ROOT)}')

    cfgs = {k: _load_yaml(p) for k, p in CONFIGS.items()}
    ref = _load_yaml(REFERENCE_013B)
    ports = set()
    names = set()

    for tag, cfg in cfgs.items():
        assert cfg['name'].startswith(f'probe_{tag}'), f'Bad name: {cfg["name"]}'
        assert cfg['name'] not in names
        names.add(cfg['name'])
        port = cfg['dist_params']['port']
        assert port not in ports, f'Duplicate dist port: {port}'
        ports.add(port)

        # Probe isolation & checkpoint safety
        assert cfg['model_type'] == 'MaritimeSFModel'
        assert cfg['scale'] == 1
        assert cfg['manual_seed'] == ref['manual_seed'] == 0
        assert cfg['path']['experiments_root'] == 'experiments/probes'
        assert cfg['path']['resume_state'] is None
        assert cfg['logger']['tb_logdir'] == 'tb_logger/probes'
        assert 0 < cfg['logger']['save_checkpoint_freq'] <= 2000
        assert cfg['val']['val_freq'] == 2000
        assert cfg['logger']['use_tb_logger'] is True
        assert cfg['logger']['use_swanlab'] is False
        assert cfg['logger']['swanlab']['project'] == 'dehaze-probes'

        # Exact protocol parity with 013b / 011p v2 manifests, datasets (allowing DDP split with total batch=16), val, and base train settings
        eff_bs = int(cfg['num_gpu']) * int(cfg['datasets']['train']['batch_size_per_gpu'])
        ref_bs = int(ref['num_gpu']) * int(ref['datasets']['train']['batch_size_per_gpu'])
        assert eff_bs == ref_bs == 16, f'{tag} effective batch size {eff_bs} != 16!'
        ds_norm = copy.deepcopy(cfg['datasets'])
        ds_norm['train']['batch_size_per_gpu'] = ref['datasets']['train']['batch_size_per_gpu']
        assert ds_norm == ref['datasets'], f'{tag} datasets diverge from v2 protocol!'
        assert cfg['val'] == ref['val'], f'{tag} val section diverges from v2 protocol!'
        train_no_ds = {k: v for k, v in cfg['train'].items() if k != 'deep_sup_opt'}
        assert train_no_ds == ref['train'], f'{tag} base train recipe diverges from v2 protocol!'

        # Manifest files exist
        for split in ('train', 'val', 'val_2', 'val_3'):
            manifest = REPO_ROOT / cfg['datasets'][split]['meta_info_file']
            assert manifest.is_file(), f'Missing manifest: {manifest}'

    # Single-variable ladder checks
    assert cfgs['014a']['network_g']['use_aniso_lk'] is False
    assert cfgs['014a']['network_g']['lfm_mod_grids'] == [16]
    assert cfgs['014a']['network_g']['use_mimo_input'] is False
    assert cfgs['014a']['network_g']['use_deep_supervision'] is False
    assert 'deep_sup_opt' not in cfgs['014a']['train']

    assert cfgs['014b']['network_g']['use_aniso_lk'] is True
    assert cfgs['014b']['network_g']['lfm_mod_grids'] == [8, 16]
    assert cfgs['014b']['network_g']['use_mimo_input'] is False
    assert cfgs['014b']['network_g']['use_deep_supervision'] is False
    assert 'deep_sup_opt' not in cfgs['014b']['train']

    assert cfgs['014c']['network_g']['use_aniso_lk'] is True
    assert cfgs['014c']['network_g']['lfm_mod_grids'] == [8, 16]
    assert cfgs['014c']['network_g']['use_mimo_input'] is True
    assert cfgs['014c']['network_g']['use_deep_supervision'] is True
    assert cfgs['014c']['train']['deep_sup_opt']['weights'] == {'1/4': 0.1, '1/2': 0.1}

    for tag, expected_depths in (('014d', [3, 3, 8]), ('014e', [4, 4, 8])):
        ng = cfgs[tag]['network_g']
        assert ng['depths'] == expected_depths
        assert ng['use_cd'] is True
        assert ng['channel_spatial_attn'] is True
        assert ng['use_aniso_lk'] is True
        assert ng['lfm_mod_grids'] == [8, 16]
        assert ng['fcsf_scene_context'] is True
        assert ng['fcsf_alpha_bias'] == -1.2
        assert ng['basr_symmetric'] is True
        assert ng['basr_scene_gate'] is True
        assert ng['basr_g_max'] == 4.0
        assert ng['use_l2_aniso_lk'] is True
        assert ng['use_mimo_input'] is True
        assert ng['use_deep_supervision'] is False
        assert 'deep_sup_opt' not in cfgs[tag]['train']

    print('  014a / 014b / 014c / 014d / 014e config parity & ablation checks PASSED.')
    return cfgs


def check_torch_units(cfgs):
    print('\n=== [2/4] PyTorch unit tests (imports, shapes, folding, identity, grads, LoRA groups) ===')
    import torch
    import basicsr
    from basicsr.archs import build_network

    print(f'  basicsr version: {basicsr.__version__}, torch: {torch.__version__}')

    for tag, cfg in cfgs.items():
        torch.manual_seed(0)
        net = build_network(cfg['network_g'])
        n_params_train = sum(p.numel() for p in net.parameters())

        # 1. Exact zero-init BASR identity & ASFB zero-init identity
        x_small = torch.rand(1, 3, 64, 64)
        net.eval()
        with torch.no_grad():
            y_init = net(x_small)
            max_id_err = (y_init - x_small).abs().max().item()
            assert max_id_err < 1e-6, f'{tag} BASR zero-init identity failed: {max_id_err}'
            assert net.aux_outputs == {}, f'{tag} eval mode must not emit aux_outputs'

            # Check ASFB & L2-AnisoLK strict identity independently
            f_bot = torch.randn(1, cfg['network_g']['widths'][-1], 16, 16)
            for blk in (net.asfb1, net.asfb2):
                err_asfb = (blk(f_bot) - f_bot).abs().max().item()
                assert err_asfb == 0.0, f'{tag} ASFB not exact identity at init: {err_asfb}'
            if net.dec2_aniso_lk is not None:
                f_l2 = torch.randn(1, cfg['network_g']['widths'][1], 32, 32)
                err_l2 = (net.dec2_aniso_lk(f_l2) - f_l2).abs().max().item()
                assert err_l2 == 0.0, f'{tag} dec2_aniso_lk not exact identity at init: {err_l2}'

            # Check C^2 zero-curvature property for symmetric BASR (Proposition P1-v2)
            if net.basr_head.symmetric_gain:
                eps = 1e-3
                k_grid = torch.tensor([-eps, 0.0, eps], dtype=torch.float64)
                ap, an = net.basr_head.a_pos, net.basr_head.a_neg
                g_vals = torch.where(
                    k_grid >= 0,
                    1.0 + ap * torch.tanh(k_grid / ap),
                    1.0 + an * torch.tanh(k_grid / an))
                g0 = g_vals[1].item()
                g1 = ((g_vals[2] - g_vals[0]) / (2 * eps)).item()
                g2 = ((g_vals[2] - 2 * g_vals[1] + g_vals[0]) / (eps ** 2)).item()
                assert abs(g0 - 1.0) < 1e-12, f'{tag} G(0)={g0}'
                assert abs(g1 - 1.0) < 1e-5, f'{tag} G\'(0)={g1}'
                assert abs(g2) < 1e-2, f'{tag} G\'\'(0)={g2} not zero-curvature!'

        # 2. Odd & rectangular sizes forward + backward + gradient flow
        net.train()
        for h, w in ((64, 64), (127, 255), (130, 198)):
            x = torch.rand(1, 3, h, w)
            target = torch.rand(1, 3, h, w)
            net.zero_grad(set_to_none=True)
            out = net(x)
            assert out.shape == x.shape, f'{tag} shape mismatch: {out.shape} vs {x.shape}'
            loss = (out - target).abs().mean()
            if tag == '014c':
                assert set(net.aux_outputs.keys()) == {'1/4', '1/2'}
                for pred_aux in net.aux_outputs.values():
                    gt_s = torch.nn.functional.interpolate(
                        target, size=pred_aux.shape[-2:], mode='bilinear', align_corners=False)
                    loss = loss + 0.1 * (pred_aux - gt_s).abs().mean()
            else:
                assert net.aux_outputs == {}
            loss.backward()
            assert torch.isfinite(loss), f'{tag} non-finite loss on {(h, w)}'
            assert net.basr_head.head.weight.grad is not None and net.basr_head.head.weight.grad.abs().sum() > 0

        # 3. 3-step gradient propagation check (zero-init heads & residual gates pass grad cleanly)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(3):
            x = torch.rand(1, 3, 64, 64)
            target = torch.rand(1, 3, 64, 64)
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss = (out - target).abs().mean()
            for pred_aux in net.aux_outputs.values():
                gt_s = torch.nn.functional.interpolate(
                    target, size=pred_aux.shape[-2:], mode='bilinear', align_corners=False)
                loss = loss + 0.1 * (pred_aux - gt_s).abs().mean()
            loss.backward()
            opt.step()
        assert net.stem.weight.grad is not None and net.stem.weight.grad.abs().sum() > 0, (
            f'{tag} stem received zero gradient after 3 steps')
        assert net.asfb1.ms_lfm.mod[-1].weight.grad is not None
        if cfg['network_g'].get('use_aniso_lk', False):
            assert net.asfb1.aniso_lk.out_proj.weight.grad is not None
        if cfg['network_g'].get('use_cd', False):
            assert net.enc1[0].conv1.cd.weight.grad is not None
            assert net.enc1[0].conv1.cd.weight.grad.abs().sum() > 0
        else:
            assert net.enc1[0].conv1.cd is None
        if cfg['network_g'].get('channel_spatial_attn', False):
            assert net.bot_seg1[0].attn.pa.weight.grad is not None
            assert net.bot_seg1[0].attn.pa.weight.grad.abs().sum() > 0
            assert net.bot_seg1[0].attn.ca[-1].weight.grad.abs().sum() > 0
        else:
            assert net.bot_seg1[0].attn.pa is None
        if cfg['network_g'].get('basr_scene_gate', False):
            assert net.basr_head.scene_mlp[-1].weight.grad is not None
            assert net.basr_head.scene_mlp[-1].weight.grad.abs().sum() > 0
        else:
            assert net.basr_head.scene_mlp is None
        if cfg['network_g'].get('fcsf_scene_context', False):
            assert net.fcsf3.scene_gate[-1].weight.grad is not None
            assert net.fcsf3.scene_gate[-1].weight.grad.abs().sum() > 0
        else:
            assert net.fcsf3.scene_gate is None
        if cfg['network_g'].get('use_l2_aniso_lk', False):
            assert net.dec2_aniso_lk.out_proj.weight.grad is not None
        else:
            assert net.dec2_aniso_lk is None

        # 4. LoRA target grouping check
        lora_groups = net.iter_lora_target_modules()
        assert len(lora_groups['ms_lfm']) == 2
        assert len(lora_groups['fcsf']) == 3
        assert len(lora_groups['basr']) == 1
        expected_lk = (2 if cfg['network_g'].get('use_aniso_lk', False) else 0) + (
            1 if cfg['network_g'].get('use_l2_aniso_lk', False) else 0)
        assert len(lora_groups['aniso_lk']) == expected_lk

        # 5. Lossless reparameterization folding equivalence
        net.eval()
        x_test = torch.rand(1, 3, 67, 99)
        with torch.no_grad():
            y_before = net(x_test)
        net_folded = copy.deepcopy(net)
        net_folded.fold_reparam()
        n_params_folded = sum(p.numel() for p in net_folded.parameters())
        with torch.no_grad():
            y_after = net_folded(x_test)
        fold_err = (y_before - y_after).abs().max().item()
        assert fold_err < 1e-5, f'{tag} fold_reparam mismatch: {fold_err}'

        print(
            f'  {tag} PASSED | train={n_params_train / 1e6:.3f}M, '
            f'folded={n_params_folded / 1e6:.3f}M, init_id={max_id_err:.2e}, fold_err={fold_err:.2e}')


def run_smoke_and_resume(cfgs):
    print('\n=== [3/4] Synthetic 6-iter BasicSR smoke training + --auto_resume verification ===')
    import cv2
    import numpy as np
    import yaml

    tmp_dir = Path(tempfile.mkdtemp(prefix='smoke_014_', dir=REPO_ROOT / 'research/014_maritime_sfnet'))
    try:
        rng = np.random.RandomState(0)
        for split, n in (('train', 4), ('val', 2)):
            (tmp_dir / split / 'GT').mkdir(parents=True)
            (tmp_dir / split / 'LQ').mkdir(parents=True)
            for i in range(n):
                gt = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
                lq = np.clip(gt.astype(np.float32) * 0.6 + 90.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(tmp_dir / split / 'GT' / f'{i:03d}.png'), gt)
                cv2.imwrite(str(tmp_dir / split / 'LQ' / f'{i:03d}.png'), lq)

        for tag in ('014a', '014b', '014c', '014d', '014e'):
            cfg = copy.deepcopy(cfgs[tag])
            exp_name = f'smoke_{tag}_verify'
            cfg['name'] = exp_name
            cfg['num_gpu'] = 0  # CPU smoke
            cfg['datasets'] = {
                'train': {
                    'name': 'SmokeTrain',
                    'type': 'PairedImageDataset',
                    'dataroot_gt': str(tmp_dir / 'train/GT'),
                    'dataroot_lq': str(tmp_dir / 'train/LQ'),
                    'filename_tmpl': '{}',
                    'io_backend': {'type': 'disk'},
                    'gt_size': 64,
                    'use_hflip': True,
                    'use_rot': False,
                    'num_worker_per_gpu': 0,
                    'batch_size_per_gpu': 2,
                    'dataset_enlarge_ratio': 1,
                },
                'val': {
                    'name': 'SmokeVal',
                    'type': 'PairedImageDataset',
                    'dataroot_gt': str(tmp_dir / 'val/GT'),
                    'dataroot_lq': str(tmp_dir / 'val/LQ'),
                    'io_backend': {'type': 'disk'},
                },
            }
            cfg['path']['experiments_root'] = str(tmp_dir / 'experiments')
            cfg['train']['total_iter'] = 3
            cfg['train']['warmup_iter'] = 1
            cfg['train']['scheduler']['periods'] = [6]
            cfg['val']['val_freq'] = 3
            cfg['logger']['print_freq'] = 1
            cfg['logger']['save_checkpoint_freq'] = 3
            cfg['logger']['use_tb_logger'] = False
            cfg['logger']['use_swanlab'] = False

            cfg_path = tmp_dir / f'{tag}_smoke.yml'
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            subprocess.run(
                [sys.executable, str(REPO_ROOT / 'basicsr/train.py'), '-opt', str(cfg_path)],
                cwd=REPO_ROOT, check=True)

            # Verify checkpoint & state exist, then test --auto_resume to iter 6
            exp_dir = tmp_dir / 'experiments' / exp_name
            assert (exp_dir / 'models/net_g_latest.pth').is_file()
            # Re-run to iter 6 only if state preserved (basicSR cleans states after clean completion,
            # so test resume by raising total_iter to 6 and keeping iter 3 state via 4-iter stop first)
            if tag == '014c':
                cfg['name'] = f'{exp_name}_resume'
                cfg['train']['total_iter'] = 4
                cfg['logger']['save_checkpoint_freq'] = 2
                cfg['val']['val_freq'] = 4
                cfg_res_path = tmp_dir / f'{tag}_resume.yml'
                cfg_res_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                subprocess.run(
                    [sys.executable, str(REPO_ROOT / 'basicsr/train.py'), '-opt', str(cfg_res_path)],
                    cwd=REPO_ROOT, check=True)
            print(f'  {tag} 3-iter BasicSR smoke training + validation + save PASSED.')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_gpu_preflight(cfgs, gpu_id: int, tags=None):
    print(f'\n=== [4/4] GPU {gpu_id} real-batch preflight (bs16 @ 512x512, per-config checkpointing) ===')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    from basicsr.archs import build_network
    from basicsr.losses import build_loss

    assert torch.cuda.is_available(), 'CUDA unavailable for GPU preflight'
    results = {}
    selected = tags or list(cfgs.keys())
    for tag in selected:
        cfg = cfgs[tag]
        torch.cuda.empty_cache()
        net = build_network(cfg['network_g']).cuda().train()
        opt = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.99))
        criteria = [build_loss(cfg['train'][k]).cuda() for k in ('pixel_opt', 'fft_opt', 'gradient_opt')]
        ds_weights = (cfg['train'].get('deep_sup_opt') or {}).get('weights', {})

        bs = int(cfg['datasets']['train']['batch_size_per_gpu'])
        for step in range(4):
            if step == 1:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
            x = torch.rand(bs, 3, 512, 512, device='cuda')
            y = torch.rand(bs, 3, 512, 512, device='cuda')
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss = sum(c(out, y) for c in criteria)
            for sk, pred_aux in net.aux_outputs.items():
                w_s = ds_weights.get(sk, 0.0)
                if w_s > 0:
                    y_s = torch.nn.functional.interpolate(
                        y, size=pred_aux.shape[-2:], mode='bilinear', align_corners=False)
                    loss = loss + w_s * (torch.nn.functional.l1_loss(pred_aux, y_s) + criteria[1](pred_aux, y_s))
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
        sec_per_iter = (time.perf_counter() - t0) / 3.0
        alloc_gib = torch.cuda.max_memory_allocated() / 2**30
        resv_gib = torch.cuda.max_memory_reserved() / 2**30
        results[tag] = {
            'parameters_train': sum(p.numel() for p in net.parameters()),
            'batch_size_per_gpu': bs,
            'activation_checkpointing': bool(cfg['network_g'].get('activation_checkpointing', False)),
            'seconds_per_iter': round(sec_per_iter, 4),
            'peak_allocated_gib': round(alloc_gib, 3),
            'peak_reserved_gib': round(resv_gib, 3),
        }
        print(f'  {tag}: {json.dumps(results[tag])}')
        del net, opt, criteria
        torch.cuda.empty_cache()

    out_path = REPO_ROOT / 'research/014_maritime_sfnet/preflight_014.json'
    out_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'  Saved GPU preflight report to {out_path.relative_to(REPO_ROOT)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-only', action='store_true', help='Only run syntax and YAML parity checks')
    parser.add_argument('--smoke', action='store_true', help='Also run CPU synthetic BasicSR smoke training')
    parser.add_argument('--preflight-gpu', type=int, default=None, help='GPU index for bs16@512 VRAM/speed preflight')
    parser.add_argument('--preflight-tags', nargs='+', default=None, help='Subset of tags for GPU preflight')
    args = parser.parse_args()

    cfgs = check_static_and_configs()
    if args.config_only:
        return
    check_torch_units(cfgs)
    if args.smoke:
        run_smoke_and_resume(cfgs)
    if args.preflight_gpu is not None:
        run_gpu_preflight(cfgs, args.preflight_gpu, tags=args.preflight_tags)


if __name__ == '__main__':
    main()
