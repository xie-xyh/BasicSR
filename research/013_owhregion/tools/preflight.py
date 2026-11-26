import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from basicsr.archs import build_network
from basicsr.losses import build_loss


def main(args):
    config = yaml.safe_load(args.config.read_text())
    network = build_network(config['network_g']).cuda().train()
    optimizer = torch.optim.Adam(network.parameters(), lr=2e-4, betas=(0.9, 0.99))
    criteria = [build_loss(config['train'][key]).cuda()
                for key in ('pixel_opt', 'fft_opt', 'gradient_opt')]
    free_before, total = torch.cuda.mem_get_info()
    for step in range(args.steps + 1):
        if step == 1:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        image = torch.rand(args.batch_size, 3, args.size, args.size, device='cuda')
        target = torch.rand(args.batch_size, 3, args.size, args.size, device='cuda')
        optimizer.zero_grad(set_to_none=True)
        prediction = network(image)
        loss = sum(criterion(prediction, target) for criterion in criteria)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print(f'step {step + 1}: loss={float(loss):.6f}', flush=True)
    free_after, _ = torch.cuda.mem_get_info()
    result = {
        'config': str(args.config),
        'network': config['network_g']['type'],
        'network_options': config['network_g'],
        'parameters': sum(p.numel() for p in network.parameters()),
        'batch_size': args.batch_size,
        'image_size': [args.size, args.size],
        'warmup_steps': 1,
        'timed_steps': args.steps,
        'seconds_per_iter': (time.perf_counter() - start) / args.steps,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'free_before_gib': free_before / 2**30,
        'free_after_gib': free_after / 2**30,
        'total_gib': total / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if args.max_reserved_gib is not None:
        assert result['peak_reserved_gib'] <= args.max_reserved_gib
        assert result['peak_reserved_gib'] <= result['free_before_gib'] - args.reserve_gib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--max-reserved-gib', type=float)
    parser.add_argument('--reserve-gib', type=float, default=2.0)
    main(parser.parse_args())
