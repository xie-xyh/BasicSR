import argparse
import json
from pathlib import Path
import time

import torch
import yaml

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.models import build_model
from basicsr.utils import set_random_seed


def benchmark(args):
    config = yaml.safe_load(args.config.read_text())
    config.update(is_train=True, dist=False, rank=0, world_size=1)
    options = config['datasets']['train']
    options.update(phase='train', scale=config['scale'], batch_size_per_gpu=args.batch)
    if args.patch:
        options['gt_size'] = args.patch
    seed = config['manual_seed']
    set_random_seed(seed)
    torch.backends.cudnn.benchmark = True
    dataset = build_dataset(options)
    sampler = EnlargedSampler(dataset, 1, 0, options.get('dataset_enlarge_ratio', 1))
    loader = build_dataloader(dataset, options, num_gpu=1, dist=False, sampler=sampler, seed=seed)
    model = build_model(config)
    batches = iter(loader)
    warmup = args.warmup
    for step in range(1, warmup + args.steps + 1):
        if step == warmup + 1:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        model.update_learning_rate(step, warmup_iter=config['train']['warmup_iter'])
        model.feed_data(next(batches))
        model.optimize_parameters(step)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return dict(variant=config['name'], batch=args.batch, patch=options['gt_size'], steps=args.steps,
                images_per_second=args.batch * args.steps / elapsed, seconds_per_step=elapsed / args.steps,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('batch', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--patch', type=int)
    args = parser.parse_args()
    report = benchmark(args)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))
