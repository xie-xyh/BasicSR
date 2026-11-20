import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys


ITER = re.compile(r'iter:\s*([\d,]+), lr:')
START = re.compile(r'Start training from epoch: \d+, iter: (\d+)')
VAL = re.compile(r'INFO: Validation (.+)$')
METRIC = re.compile(r'# (psnr|ssim):\s+([\w.+-]+)')


def parse(log):
    iteration = 0
    dataset = None
    scores = {}
    for line in log.read_text(errors='replace').splitlines():
        step = ITER.search(line) or START.search(line)
        if step:
            iteration = int(step[1].replace(',', ''))
            dataset = None
        validation = VAL.search(line)
        if validation:
            dataset = validation[1].strip()
        metric = METRIC.search(line)
        if metric and dataset:
            scores.setdefault((iteration, dataset), {})[metric[1]] = float(metric[2])
    return iteration, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.paths:
        log = path / 'console.log' if path.is_dir() else path
        iteration, scores = parse(log)
        status = path / 'exit_code'
        pid_file = path / 'pid'
        if status.is_file():
            state = f'退出码 {status.read_text().strip()}'
        elif pid_file.is_file():
            pid = pid_file.read_text().strip()
            process = subprocess.run(['ps', '-p', pid, '-o', 'args='], capture_output=True, text=True)
            state = f'运行中 (PID {pid})' if 'basicsr/train.py' in process.stdout else '进程已退出，未写入退出码'
        else:
            state = '日志快照'
        print(f'{path.name}: {state}，最新记录 {iteration:,} 步', file=sys.stderr)
        rows.extend(dict(run=path.name, iteration=step, dataset=dataset, **metrics)
                    for (step, dataset), metrics in sorted(scores.items()))
    if args.csv:
        with args.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('run', 'iteration', 'dataset', 'psnr', 'ssim'))
            writer.writeheader()
            writer.writerows(rows)
    else:
        for row in rows:
            print(row)


if __name__ == '__main__':
    main()
