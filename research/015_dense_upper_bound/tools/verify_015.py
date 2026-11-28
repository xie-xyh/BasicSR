"""Unit, backward-compatibility, synthetic smoke, and 3090 VRAM preflight tests for 015."""
import argparse
import copy
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {
    '015a': REPO_ROOT / 'research/015_dense_upper_bound/probe_015a_axial_basr.yml',
    '015b': REPO_ROOT / 'research/015_dense_upper_bound/probe_015b_maritime_recipe.yml',
    '015c': REPO_ROOT / 'research/015_dense_upper_bound/probe_015c_axial_basr_maritime.yml',
    '015d': REPO_ROOT / 'research/015_dense_upper_bound/probe_015d_norot_only.yml',
}
REF_014D = REPO_ROOT / 'research/014_maritime_sfnet/probe_014d.yml'


def _load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text())


def check_static_and_configs():
    print('=== [1/4] Syntax & 015 YAML orthogonality audit ===')
    for src in (
        REPO_ROOT / 'basicsr/archs/maritime_sfnet_arch.py',
        REPO_ROOT / 'basicsr/models/maritime_sf_model.py',
    ):
        py_compile.compile(str(src), doraise=True)
        print(f'  syntax OK: {src.relative_to(REPO_ROOT)}')

    cfgs = {k: _load_yaml(p) for k, p in CONFIGS.items()}
    ref = _load_yaml(REF_014D)
    ports = {ref['dist_params']['port']}
    names = {ref['name']}

    for tag, cfg in cfgs.items():
        assert cfg['name'].startswith(f'probe_{tag}'), f'Bad name: {cfg["name"]}'
        assert cfg['name'] not in names
        names.add(cfg['name'])
        port = cfg['dist_params']['port']
        assert port not in ports, f'Duplicate port: {port}'
        ports.add(port)

        assert cfg['model_type'] == 'MaritimeSFModel'
        assert cfg['scale'] == 1
        assert cfg['manual_seed'] == 0
        eff_bs = int(cfg['num_gpu']) * int(cfg['datasets']['train']['batch_size_per_gpu'])
        assert eff_bs == 16, f'{tag} eff_bs={eff_bs} != 16'
        assert cfg['path']['experiments_root'] == 'experiments/probes'
        assert cfg['path']['resume_state'] is None
        assert cfg['logger']['tb_logdir'] == 'tb_logger/probes'
        assert cfg['logger']['use_tb_logger'] is True
        assert cfg['logger']['use_swanlab'] is False
        assert cfg['logger']['swanlab']['project'] == 'dehaze-probes'
        assert 0 < cfg['logger']['save_checkpoint_freq'] <= 2000
        assert cfg['val'] == ref['val'], f'{tag} val section diverges from 014d!'
        for split in ('train', 'val', 'val_2', 'val_3'):
            manifest = REPO_ROOT / cfg['datasets'][split]['meta_info_file']
            assert manifest.is_file(), f'Missing manifest: {manifest}'

    # 015a: strict v2 data & loss parity with 014d, architecture-only changes
    c15a = cfgs['015a']
    assert c15a['datasets']['train']['meta_info_file'] == ref['datasets']['train']['meta_info_file']
    assert c15a['datasets']['train']['use_rot'] is True
    assert c15a['train'] == ref['train']
    assert c15a['network_g']['use_axial_global'] is True
    assert c15a['network_g']['basr_g_max'] == 6.0
    assert c15a['network_g']['basr_refine'] is True

    # 015b: strict 014d architecture parity, maritime dense-focused recipe changes only
    c15b = cfgs['015b']
    ng_15b = {k: v for k, v in c15b['network_g'].items() if k not in ('use_axial_global', 'basr_refine')}
    assert ng_15b == ref['network_g'], '015b architecture diverges from 014d!'
    assert c15b['network_g']['use_axial_global'] is False
    assert c15b['network_g']['basr_refine'] is False
    assert c15b['datasets']['train']['use_rot'] is False
    assert c15b['datasets']['train']['use_hflip'] is True
    assert c15b['train']['fft_opt']['type'] == 'BandBalancedFFTLoss'
    assert c15b['train']['fft_opt']['low_weight'] == 0.00607

    # 015c: exact combination of 015a architecture + 015b maritime recipe
    c15c = cfgs['015c']
    assert c15c['network_g'] == c15a['network_g'], '015c architecture must match 015a!'
    assert c15c['datasets'] == c15b['datasets'], '015c datasets must match 015b!'
    assert c15c['train'] == c15b['train'], '015c train recipe must match 015b!'

    # 015d: strict single-variable ablation of use_rot=false on 014d (exact v2 manifest + FFTLoss)
    c15d = cfgs['015d']
    assert c15d['network_g'] == c15b['network_g'], '015d architecture must match 014d/015b!'
    assert c15d['train'] == ref['train'], '015d train losses/schedule must match 014d!'
    assert c15d['datasets']['train']['meta_info_file'] == ref['datasets']['train']['meta_info_file']
    assert c15d['datasets']['train']['use_rot'] is False
    assert c15d['datasets']['train']['use_hflip'] is True

    # Verify dense322 manifest has zero leakage (exact subset of v2 train manifest)
    base_lines = [l.strip() for l in (REPO_ROOT / ref['datasets']['train']['meta_info_file']).read_text().splitlines() if l.strip()]
    d322_lines = [l.strip() for l in (REPO_ROOT / c15b['datasets']['train']['meta_info_file']).read_text().splitlines() if l.strip()]
    assert set(d322_lines) == set(base_lines), 'dense322 manifest must cover exact same unique images as v2 train!'
    print(f'  dense322 manifest verified: {len(d322_lines)} rows from {len(base_lines)} unique v2 train pairs.')
    print('  015a / 015b / 015c / 015d factorial orthogonality & isolation checks PASSED.')
    return cfgs, ref


def check_torch_units(cfgs, ref):
    print('\n=== [2/4] PyTorch unit, 014d strict compatibility, identity, gradient & folding tests ===')
    import torch
    import basicsr
    from basicsr.archs import build_network

    print(f'  basicsr={basicsr.__version__}, torch={torch.__version__}')

    # 1. Backward compatibility with existing 014d checkpoint if present on server
    net_14d = build_network(ref['network_g'])
    net_15b = build_network(cfgs['015b']['network_g'])
    assert list(net_14d.state_dict().keys()) == list(net_15b.state_dict().keys()), (
        '015b state_dict keys differ from 014d!')
    ckpt_14d = REPO_ROOT / 'experiments/probes/probe_014d_hust_v2_8000iter/models/net_g_latest.pth'
    if ckpt_14d.is_file():
        sd = torch.load(ckpt_14d, map_location='cpu')
        net_14d.load_state_dict(sd['params_ema'], strict=True)
        print(f'  Strict load of live 014d EMA checkpoint PASSED: {ckpt_14d.name}')

    for tag, cfg in cfgs.items():
        torch.manual_seed(0)
        net = build_network(cfg['network_g'])
        n_train = sum(p.numel() for p in net.parameters())

        # Zero-init identity checks
        x_small = torch.rand(1, 3, 64, 64)
        net.eval()
        with torch.no_grad():
            y_init = net(x_small)
            id_err = (y_init - x_small).abs().max().item()
            assert id_err < 1e-6, f'{tag} zero-init identity failed: {id_err}'
            f_bot = torch.randn(1, cfg['network_g']['widths'][-1], 16, 16)
            for blk in (net.asfb1, net.asfb2):
                assert (blk(f_bot) - f_bot).abs().max().item() == 0.0
                if blk.axial_global is not None:
                    assert (blk.axial_global(f_bot) - f_bot).abs().max().item() == 0.0

        # Odd & rectangular shapes + finite grads + 3-step gradient flow into new modules
        net.train()
        for h, w in ((64, 64), (127, 255), (130, 198)):
            x = torch.rand(1, 3, h, w)
            t = torch.rand(1, 3, h, w)
            net.zero_grad(set_to_none=True)
            out = net(x)
            assert out.shape == x.shape
            loss = (out - t).abs().mean()
            loss.backward()
            assert torch.isfinite(loss)

        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(3):
            x = torch.rand(1, 3, 64, 64)
            t = torch.rand(1, 3, 64, 64)
            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss = (out - t).abs().mean()
            loss.backward()
            opt.step()

        assert net.stem.weight.grad is not None and net.stem.weight.grad.abs().sum() > 0
        assert net.diagnostics['diag_basr_gain_p95'] is not None
        if cfg['network_g'].get('use_axial_global', False):
            assert net.asfb1.axial_global.row_net[0].weight.grad.abs().sum() > 0
            assert net.asfb1.axial_global.col_net[0].weight.grad.abs().sum() > 0
            assert net.basr_head.refine[0].weight.grad.abs().sum() > 0
            assert net.diagnostics['diag_basr_refine'] is not None
            assert len(net.iter_lora_target_modules()['axial_global']) == 2
        else:
            assert net.asfb1.axial_global is None
            assert net.basr_head.refine is None
            assert len(net.iter_lora_target_modules()['axial_global']) == 0

        # Lossless folding check
        net.eval()
        x_test = torch.rand(1, 3, 67, 99)
        with torch.no_grad():
            y_before = net(x_test)
        net_f = copy.deepcopy(net)
        net_f.fold_reparam()
        n_fold = sum(p.numel() for p in net_f.parameters())
        with torch.no_grad():
            y_after = net_f(x_test)
        fold_err = (y_before - y_after).abs().max().item()
        assert fold_err < 1e-5, f'{tag} fold error {fold_err}'
        print(f'  {tag} PASSED | train={n_train/1e6:.3f}M, folded={n_fold/1e6:.3f}M, init_id={id_err:.2e}, fold_err={fold_err:.2e}')


def run_smoke(cfgs):
    print('\n=== [3/4] Synthetic 3-iter BasicSR full-pipeline smoke training ===')
    import cv2
    import numpy as np
    import yaml

    tmp_dir = Path(tempfile.mkdtemp(prefix='smoke_015_', dir=REPO_ROOT / 'research/015_dense_upper_bound'))
    try:
        rng = np.random.RandomState(0)
        for split, n in (('train', 4), ('val', 2)):
            (tmp_dir / split / 'GT').mkdir(parents=True)
            (tmp_dir / split / 'LQ').mkdir(parents=True)
            for i in range(n):
                gt = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
                lq = np.clip(gt.astype(np.float32) * 0.5 + 110.0, 0, 255).astype(np.uint8)
                cv2.imwrite(str(tmp_dir / split / 'GT' / f'{i:03d}.png'), gt)
                cv2.imwrite(str(tmp_dir / split / 'LQ' / f'{i:03d}.png'), lq)

        for tag, orig in cfgs.items():
            cfg = copy.deepcopy(orig)
            cfg['name'] = f'smoke_{tag}_verify'
            cfg['num_gpu'] = 0
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
                    'use_rot': cfg['datasets']['train']['use_rot'],
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
            assert (tmp_dir / 'experiments' / cfg['name'] / 'models/net_g_latest.pth').is_file()
            print(f'  {tag} 3-iter BasicSR smoke training PASSED.')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_gpu_preflight(cfgs, gpu_id: int):
    print(f'\n=== [4/4] GPU {gpu_id} real-batch preflight (bs16 @ 512x512, ckpt=True) ===')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    from basicsr.archs import build_network
    from basicsr.losses import build_loss

    assert torch.cuda.is_available(), 'CUDA unavailable'
    results = {}
    for tag, cfg in cfgs.items():
        torch.cuda.empty_cache()
        net = build_network(cfg['network_g']).cuda().train()
        opt = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.99))
        criteria = [build_loss(cfg['train'][k]).cuda() for k in ('pixel_opt', 'fft_opt', 'gradient_opt')]
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
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / 3.0
        alloc = torch.cuda.max_memory_allocated() / 2**30
        resv = torch.cuda.max_memory_reserved() / 2**30
        results[tag] = {
            'params_train': sum(p.numel() for p in net.parameters()),
            'batch_size_per_gpu': bs,
            'seconds_per_iter': round(sec, 4),
            'peak_allocated_gib': round(alloc, 3),
            'peak_reserved_gib': round(resv, 3),
        }
        print(f'  {tag}: {json.dumps(results[tag])}')
        del net, opt, criteria
        torch.cuda.empty_cache()

    out_path = REPO_ROOT / 'research/015_dense_upper_bound/preflight_015.json'
    out_path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'  Saved preflight report to {out_path.relative_to(REPO_ROOT)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight-gpu', type=int, default=None)
    args = parser.parse_args()

    cfgs, ref = check_static_and_configs()
    if args.config_only:
        return
    check_torch_units(cfgs, ref)
    if args.smoke:
        run_smoke(cfgs)
    if args.preflight_gpu is not None:
        run_gpu_preflight(cfgs, args.preflight_gpu)


if __name__ == '__main__':
    main()
