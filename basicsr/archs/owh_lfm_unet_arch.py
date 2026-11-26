"""Four-scale paired RGB dehazing model for BasicSR."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from basicsr.archs.deanet_v3_arch import LowFreqModBlock
from basicsr.utils.registry import ARCH_REGISTRY


class RestorationBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.expand = nn.Conv2d(channels, 3 * channels, 1)
        self.local = nn.Conv2d(2 * channels, 2 * channels, kernel,
                               padding=kernel // 2, groups=2 * channels)
        self.project = nn.Conv2d(2 * channels, channels, 1)
        self.scale = nn.Parameter(torch.full((channels, 1, 1), 0.1))

    def forward(self, x):
        local, context = self.expand(x).split([2 * x.shape[1], x.shape[1]], dim=1)
        a, b = self.local(local).chunk(2, dim=1)
        update = torch.cat((F.gelu(a), b * torch.sigmoid(context)), dim=1)
        return x + self.scale * self.project(update)


class FullConvRestorationBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1))
        self.scale = nn.Parameter(torch.full((channels, 1, 1), 0.1))

    def forward(self, x):
        return x + self.scale * self.spatial(x)


class LowFrequencyRGBHead(nn.Module):
    def __init__(self, channels, grid_size=16, hidden=64):
        super().__init__()
        self.grid_size = grid_size
        self.predict = nn.Sequential(
            nn.Conv2d(channels + 3, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 3, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, features, image):
        size = (self.grid_size, self.grid_size)
        evidence = torch.cat((F.adaptive_avg_pool2d(features, size),
                              F.adaptive_avg_pool2d(image, size)), dim=1)
        correction = self.predict(evidence)
        return F.interpolate(correction, size=image.shape[-2:], mode='bilinear', align_corners=False)


def stage(channels, depth):
    return nn.Sequential(*(RestorationBlock(channels, 3 if i % 2 == 0 else 5)
                           for i in range(depth)))


class BandFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weights = nn.Sequential(
            nn.Conv2d(2 * channels, channels // 2, 1), nn.GELU(),
            nn.Conv2d(channels // 2, 2 * channels, 1))
        nn.init.zeros_(self.weights[-1].weight)
        nn.init.zeros_(self.weights[-1].bias)

    def forward(self, deep, skip):
        smooth = F.avg_pool2d(F.pad(skip, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
        evidence = torch.cat((deep, smooth - deep), dim=1)
        low, detail = (2 * torch.sigmoid(self.weights(evidence))).chunk(2, dim=1)
        return deep + low * smooth + detail * (skip - smooth)


class DetailRefinement(nn.Module):
    """Refine half-resolution decoder features using their local contrast."""
    def __init__(self, channels, mode):
        super().__init__()
        if mode not in ("strip", "square"):
            raise ValueError("detail_mode must be None, 'strip', or 'square'")
        hidden = channels // 2
        self.project = nn.Conv2d(channels, 2 * hidden, 1, bias=False)
        kernels = [(1, 9), (9, 1)] if mode == "strip" else [(3, 3), (3, 3)]
        self.filters = nn.ModuleList(
            nn.Conv2d(hidden, hidden, (h, w), padding=(h // 2, w // 2),
                      padding_mode="replicate", groups=hidden, bias=False)
            for h, w in kernels)
        self.gate = nn.Conv2d(channels, 2 * hidden, 1)
        self.out = nn.Conv2d(2 * hidden, channels, 1, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x):
        smooth = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        first, second = self.project(x - smooth).chunk(2, dim=1)
        response = torch.cat((self.filters[0](first), self.filters[1](second)), dim=1)
        return x + self.out(response * torch.sigmoid(self.gate(x)))


@ARCH_REGISTRY.register()
class OWHLFMUNet(nn.Module):
    def __init__(self, widths=(40, 80, 144, 224), depths=(2, 3, 4), detail_mode=None, activation_checkpointing=False):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.stem = nn.Conv2d(3, widths[0], 3, padding=1)
        self.enc = nn.ModuleList(stage(c, n) for c, n in zip(widths, depths))
        self.down = nn.ModuleList(nn.Conv2d(a, b, 4, stride=2, padding=1)
                                  for a, b in zip(widths, widths[1:]))
        c = widths[-1]
        self.middle = nn.Sequential(stage(c, 2), LowFreqModBlock(c),
                                    stage(c, 2), LowFreqModBlock(c), stage(c, 2))
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(len(depths))):
            c = widths[i]
            self.up.append(nn.Sequential(nn.Conv2d(widths[i + 1], 4 * c, 1), nn.PixelShuffle(2)))
            self.fuse.append(BandFusion(c))
            self.dec.append(stage(c, depths[i]))
        if detail_mode is not None:
            self.dec[1].add_module("detail", DetailRefinement(widths[1], detail_mode))
        self.rgb = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.low_frequency = None

    def run_stage(self, module, *inputs):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, image):
        h, w = image.shape[-2:]
        padded = F.pad(image, (0, -w % 8, 0, -h % 8), mode='replicate')
        x = self.stem(padded)
        skips = []
        for encode, down in zip(self.enc, self.down):
            skip = self.run_stage(encode, x)
            skips.append(skip)
            x = down(skip)
        x = self.run_stage(self.middle, x)
        for up, fuse, decode, skip in zip(self.up, self.fuse, self.dec, reversed(skips)):
            x = self.run_stage(decode, self.run_stage(fuse, up(x), skip))
        correction = self.rgb(x)
        if self.low_frequency is not None:
            correction = correction + self.low_frequency(x, padded)
        return (padded + correction)[:, :, :h, :w]


class RegionContext(nn.Module):
    """Each bottleneck position reads a compact grid of regional descriptors."""
    def __init__(self, channels):
        super().__init__()
        self.heads = 8
        self.query = nn.Conv2d(channels, channels, 1)
        self.key_value = nn.Conv2d(channels, channels * 2, 1)
        self.project = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.full((channels, 1, 1), 0.01))

    def tokens(self, tensor):
        batch, channels, height, width = tensor.shape
        return tensor.reshape(batch, self.heads, channels // self.heads,
                              height * width).transpose(-2, -1)

    def forward(self, x):
        h, w = x.shape[-2:]
        grid = F.adaptive_avg_pool2d(x, (min(h, 8), min(w, 8)))
        key, value = self.key_value(grid).chunk(2, dim=1)
        query = self.tokens(self.query(x))
        scores = (query * query.shape[-1] ** -0.5) @ self.tokens(key).transpose(-2, -1)
        context = scores.softmax(dim=-1) @ self.tokens(value)
        context = context.transpose(-2, -1).reshape_as(x)
        return x + self.scale * self.project(context)


class HazeConditionedFusion(nn.Module):
    """Predict independent skip gains from regional disagreement and image context."""
    def __init__(self, channels):
        super().__init__()
        hidden = channels // 2
        self.align = nn.Conv2d(channels, channels, 1)
        self.region = nn.Sequential(nn.Conv2d(3 * channels, hidden, 1), nn.GELU(),
                                    nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden))
        self.scene = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, 1),
                                   nn.GELU(), nn.Conv2d(hidden, hidden, 1))
        self.gains = nn.Conv2d(hidden, 2 * channels, 1)
        nn.init.zeros_(self.gains.weight)
        nn.init.zeros_(self.gains.bias)

    def forward(self, deep, skip):
        aligned = self.align(skip)
        smooth = F.avg_pool2d(F.pad(aligned, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
        detail = aligned - smooth
        def pool(x):
            return F.avg_pool2d(x, 2, ceil_mode=True)
        region = self.region(torch.cat((pool(deep), pool(smooth - deep), pool(detail.abs())), 1))
        logits = self.gains(F.gelu(region + self.scene(region)))
        logits = F.interpolate(logits, size=deep.shape[-2:], mode="bilinear", align_corners=False)
        low_gain, detail_gain = (2 * logits.sigmoid()).chunk(2, dim=1)
        return deep + low_gain * smooth + detail_gain * detail


@ARCH_REGISTRY.register()
class OWHRegionUNet(OWHLFMUNet):
    """Single OWH design with regional attention, LFM and scene-conditioned skips."""
    def __init__(self, widths=(48, 96, 176, 320), depths=(2, 3, 4), detail_mode="strip",
                 activation_checkpointing=False, spatial_expression=False, low_frequency_head=False):
        super().__init__(widths=widths, depths=depths, detail_mode=detail_mode,
                         activation_checkpointing=activation_checkpointing)
        self.fuse = nn.ModuleList(HazeConditionedFusion(c) for c in reversed(widths[:-1]))
        layers = list(self.middle.children())
        layers.insert(2, RegionContext(widths[-1]))
        self.middle = nn.Sequential(*layers)
        if spatial_expression:
            self.middle[-1][-1] = FullConvRestorationBlock(widths[-1])
            self.dec[0][-1] = FullConvRestorationBlock(widths[-2])
        if low_frequency_head:
            self.low_frequency = LowFrequencyRGBHead(widths[0])
