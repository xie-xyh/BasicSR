#!/usr/bin/env python
"""006a 训练效果监控：解析 nohup 日志，落盘指标 CSV 并打印趋势摘要。

用法：
    python research/006_hihaze/tools/monitor_006a.py

行为：
- 解析 experiments/006a_nohup.log 的验证块（三子集 psnr/ssim）与训练 loss 行；
- 新增验证记录幂等追加到 research/006_hihaze/metrics_006a.csv（按 dataset+iter 去重）；
- 打印：最新 iter/速度/ETA、近期 l_sda 均值、各子集最新指标及与 004a 基线的差值、
  停滞/异常信号（日志 15 分钟未更新、loss NaN、traceback）；
- 退出码恒为 0，供 cron 巡检安全调用。
"""

import argparse
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 004a 最终全量评估基线（params_ema）
BASELINE = {
    'OWH-Dense-test': (24.8601, 0.9006),
    'OWH-Mist-test': (30.98, 0.9725),
    'OWH-Moderate-test': (28.02, 0.9411),
}

RE_ITER = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ INFO: .*iter:\s+([\d,]+), lr:\([\d.e,+-]+\)\].*?'
    r'time \(data\): ([\d.]+)')
RE_LOSS = re.compile(r'(l_\w+): ([\d.eE+-]+)')
RE_VAL = re.compile(r'INFO: Validation (\S+)')
RE_METRIC = re.compile(r'# (\w+): ([\d.]+)\s+Best: [\d.]+ @ (\d+) iter')


def parse(log_path):
    iters, vals = [], {}  # vals: dataset -> {iter: {metric: value}}
    cur_ds = None
    last_iter = None  # 验证块前的最后一个训练 iter，作为该验证的迭代数
    with open(log_path) as f:
        for line in f:
            m = RE_ITER.match(line)
            if m:
                last_iter = int(m.group(2).replace(',', ''))
                losses = {k: float(v) for k, v in RE_LOSS.findall(line)}
                iters.append(dict(ts=m.group(1), iter=last_iter,
                                  sec=float(m.group(3)), losses=losses))
                continue
            m = RE_VAL.search(line)
            if m:
                cur_ds = m.group(1)
                vals.setdefault(cur_ds, {})
                continue
            m = RE_METRIC.search(line)
            if m and cur_ds and last_iter is not None:
                vals[cur_ds].setdefault(last_iter, {})[m.group(1)] = float(m.group(2))
    return iters, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=os.path.join(ROOT, 'experiments/006a_nohup.log'))
    parser.add_argument('--csv', default=os.path.join(ROOT, 'research/006_hihaze/metrics_006a.csv'))
    parser.add_argument('--total', type=int, default=20000)
    parser.add_argument('--tag', default='006a', help='仅用于摘要展示与告警文本')
    args = parser.parse_args()
    LOG, CSV = args.log, args.csv
    if not os.path.exists(LOG):
        print(f'[monitor] 日志不存在: {LOG}')
        return
    iters, vals = parse(LOG)

    # --- 幂等追加 CSV ---
    existing = set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    existing.add((parts[0], parts[1]))
    new_rows = []
    for ds, recs in vals.items():
        for it, metrics in sorted(recs.items()):
            if (ds, str(it)) not in existing and 'psnr' in metrics and 'ssim' in metrics:
                new_rows.append((ds, it, metrics['psnr'], metrics['ssim']))
    if new_rows:
        os.makedirs(os.path.dirname(CSV), exist_ok=True)
        write_header = not os.path.exists(CSV)
        with open(CSV, 'a') as f:
            if write_header:
                f.write('dataset,iter,psnr,ssim\n')
            for ds, it, psnr, ssim in new_rows:
                f.write(f'{ds},{it},{psnr:.4f},{ssim:.4f}\n')

    # --- 摘要 ---
    stall_min = (time.time() - os.path.getmtime(LOG)) / 60
    if iters:
        last = iters[-1]
        recent = iters[-5:]
        spd = sum(r['sec'] for r in iters[-10:]) / min(10, len(iters))
        loss_str = ' '.join(
            f'{k}~{sum(r["losses"].get(k, 0) for r in recent) / len(recent):.4f}'
            for k in recent[-1]['losses'])
        print(f'[monitor:{args.tag}] iter {last["iter"]}/{args.total} | {spd:.2f} s/iter | '
              f'{loss_str} | 日志 {stall_min:.0f} 分钟前更新')
    else:
        print(f'[monitor] 尚无训练行 | 日志 {stall_min:.0f} 分钟前更新')

    for ds, (bpsnr, bssim) in BASELINE.items():
        recs = {it: m for it, m in vals.get(ds, {}).items()
                if 'psnr' in m and 'ssim' in m}  # 验证写入中途可能只有半个记录
        if not recs:
            print(f'[monitor] {ds}: 尚无验证')
            continue
        it = max(recs)
        p, s = recs[it]['psnr'], recs[it]['ssim']
        print(f'[monitor] {ds} @{it}: PSNR {p:.4f} ({p - bpsnr:+.4f} vs 004a) '
              f'SSIM {s:.4f} ({s - bssim:+.4f})')
    if new_rows:
        print(f'[monitor:{args.tag}] 新增 {len(new_rows)} 条验证记录 → {os.path.basename(CSV)}')

    # --- 异常信号 ---
    with open(LOG, errors='ignore') as f:
        tail = f.read()[-20000:]
    for sig in ('Traceback', 'CUDA out of memory', 'nan', 'NaN'):
        if sig in tail:
            print(f'[monitor][告警] 日志尾部出现 {sig!r}')
    if stall_min > 15:
        print('[monitor][告警] 日志超过 15 分钟未更新，训练可能停滞')


if __name__ == '__main__':
    sys.exit(main())
