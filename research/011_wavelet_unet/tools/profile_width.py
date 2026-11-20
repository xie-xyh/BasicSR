import argparse
import time

import torch
from torch.nn import functional as F

from basicsr.archs.wavelet_unet_arch import WaveletUNet


def profile(widths):
    torch.manual_seed(0)
    model = WaveletUNet(widths=widths).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))
    hazy = torch.rand(20, 3, 512, 512, device='cuda')
    clear = torch.rand_like(hazy)
    target_ll, target_detail = model._split_subbands(model.dwt1(clear))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for _ in range(4):
        optimizer.zero_grad()
        prediction = model(hazy)
        approx, detail = model.aux
        spectrum = torch.fft.rfft2(prediction - clear)
        loss = (F.l1_loss(prediction, clear) + spectrum.abs().mean() / 512
                + F.l1_loss(approx, target_ll) + F.l1_loss(detail, target_detail))
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss).item()
    torch.cuda.synchronize()
    print(f'widths={widths}, parameters={sum(p.numel() for p in model.parameters()):,}, '
          f'peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}, '
          f'peak_reserved_gib={torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}, '
          f'mean_step_seconds={(time.monotonic() - started) / 4:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('widths', nargs=4, type=int)
    profile(parser.parse_args().widths)
