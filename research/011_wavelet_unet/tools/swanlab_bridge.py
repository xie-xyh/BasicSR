import argparse
from collections import defaultdict
from pathlib import Path
import time

import swanlab
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml


def collect(accumulator, steps):
    accumulator.Reload()
    events = defaultdict(dict)
    for tag in accumulator.Tags()['scalars']:
        for event in accumulator.Scalars(tag):
            if event.step > steps.get(tag, -1):
                events[event.step][tag] = event.value
    return events


def sync(args):
    config_path = args.config or args.run / f'{args.arm}_train.yml'
    config = yaml.safe_load(config_path.read_text())
    project = (config['logger'].get('swanlab') or {}).get('project') or 'dehaze'
    events = args.tensorboard / config['name']
    terminal = args.exit_file or args.run / 'run' / f'{args.arm}.exit'
    deadline = time.monotonic() + args.startup_timeout
    while not any(events.glob('*tfevents*')):
        if terminal.is_file() or time.monotonic() >= deadline:
            raise FileNotFoundError(events)
        time.sleep(2)
    accumulator = EventAccumulator(str(events), size_guidance={'scalars': 0})
    steps = {}
    swanlab.init(project=project, name=config['name'], mode=args.mode, public=False,
                 log_dir=str(args.output), description='Scalar replay from the live BasicSR TensorBoard log.',
                 config={key: config[key] for key in ('network_g', 'manual_seed') if key in config}
                 | {'training_budget': config.get('train', {}).get('total_iter'),
                    'batch_size': config.get('datasets', {}).get('train', {}).get('batch_size_per_gpu')})
    try:
        while True:
            finished = terminal.is_file()
            new = collect(accumulator, steps)
            for step, metrics in sorted(new.items()):
                swanlab.log(metrics, step=step)
                steps.update(dict.fromkeys(metrics, step))
            if new:
                print(f'{config["name"]}: synced {len(new)} steps, latest={max(new)}', flush=True)
            if finished or args.once:
                break
            time.sleep(30)
    finally:
        swanlab.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('run', 'tensorboard', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('arm', choices=('control', 'candidate'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--exit-file', type=Path)
    parser.add_argument('--startup-timeout', type=int, default=300)
    parser.add_argument('--mode', choices=('online', 'local'), default='online')
    parser.add_argument('--once', action='store_true')
    sync(parser.parse_args())
