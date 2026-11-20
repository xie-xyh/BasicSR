# WaveletUNet (011a): feature-space U-Net with lossless Haar resampling +
# image-space wavelet output decomposition.
#
# Design (research/011_wavelet_unet/README.md, with full derivations):
#   - A true 4-level encoder-decoder (not a single-pass subband processor like
#     010a): capacity and multi-scale feature extraction are provided by the
#     U-Net itself; Haar WaveletDown/Up (008_dwt) give lossless resampling.
#   - The bottleneck uses pyramid pooling (1x1/4x4/8x8 grids) so the veil field
#     A(1-t) is estimated with a strictly global context.
#   - The output head performs the *image-space* physical decomposition:
#     LL1_rec = LL1 + delta (veil correction), and
#     det1_rec = det1 * gain + free_res, where gain is a bounded, smoothed
#     multiplicative field (gain ~= 1/t under locally constant t) and free_res
#     is an *unbounded* generative correction (010a's lesson: multiplication
#     alone cannot restore detail where it was destroyed).
#   - Reconstruction is exact Haar IWT back to RGB; aux exposes
#     (ll1_rec, det1_rec) for subband supervision in WaveletUNetModel.

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from .deanet_v3_arch import LowFreqModBlock
from .deanet_v6_arch import HaarDWT, HaarIWT, WaveletDown, WaveletUp


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class PyramidPool(nn.Module):
    """PSPNet-style pyramid pooling for a global veil context."""

    def __init__(self, dim, grids=(1, 4, 8)):
        super(PyramidPool, self).__init__()
        self.grids = grids
        self.proj = nn.Sequential(
            nn.Conv2d(dim * (1 + len(grids)), dim, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        feats = [x]
        for g in self.grids:
            p = F.adaptive_avg_pool2d(x, g)
            feats.append(F.interpolate(p, size=(H, W), mode='bilinear', align_corners=False))
        return x + self.proj(torch.cat(feats, dim=1))


class ContextPyramid(nn.Module):
    """Parallel depthwise convs at multiple dilations fused by a zero-init 1x1 projection.

    Used by 011d to test whether several receptive-field ranges at the
    bottleneck beat a single range at equal parameter count.
    """

    def __init__(self, dim, dilations):
        super(ContextPyramid, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim), nn.GELU())
            for dilation in dilations
        ])
        self.project = nn.Conv2d(dim * len(dilations), dim, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        return x + self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class LowBandCorrectionGate(nn.Module):
    """Content-predicted gain (±25%) on the smooth part of the LL1 correction.

    Used by 011g to let image content scale the low-frequency correction
    strength instead of keeping it fixed.
    """

    def __init__(self, channels):
        super(LowBandCorrectionGate, self).__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(channels + 3, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, correction, features, low_input):
        height, width = correction.shape[-2:]
        pooled_size = (min(height, 16), min(width, 16))
        settings = dict(size=(height, width), mode='bilinear', align_corners=False)
        signal = torch.cat((F.adaptive_avg_pool2d(features, pooled_size),
                            F.adaptive_avg_pool2d(low_input, pooled_size)), dim=1)
        deviation = 0.25 * torch.tanh(self.predict(signal))
        smooth = F.interpolate(F.adaptive_avg_pool2d(correction, pooled_size), **settings)
        return correction + F.interpolate(deviation, **settings) * smooth


class PooledResidualHead(nn.Module):
    """Predict a 3-channel residual at low resolution, bilinear-upsampled to the LL1 size.

    Shared by two probes with different feature sources and post-processing:
    coarse_color_residual (011i, bottleneck features, slow tone correction) and
    mid_band_residual (011j, 1/4 decoder features, band-passed afterwards).
    """

    def __init__(self, channels):
        super(PooledResidualHead, self).__init__()
        self.predict = nn.Sequential(
            nn.Conv2d(channels + 3, 32, 3, padding=1, padding_mode='replicate'),
            nn.GELU(), nn.Conv2d(32, 3, 1))
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

    def forward(self, features, low_input):
        reduced = F.adaptive_avg_pool2d(low_input, features.shape[-2:])
        residual = self.predict(torch.cat((features, reduced), dim=1))
        return F.interpolate(residual, size=low_input.shape[-2:], mode='bilinear', align_corners=False)


@ARCH_REGISTRY.register()
class WaveletUNet(nn.Module):
    """4-level U-Net + Haar resampling + image-space wavelet output head.

    Args:
        widths (tuple): Channels at the four encoder levels (on LL1 space).
        feat (int): Output-head feature width (decoder level-1 channels).
        use_lfm (bool): Insert a LowFreqModBlock at the bottleneck (011b).
        detail_stem (bool): Also feed the 9ch input high frequencies into the
            first encoder feature via a zero-init conv (011c).
        context_dilations (tuple | None): If set, add a ContextPyramid with
            these dilation rates at the bottleneck (011d).
        low_band_gate (bool): Add a LowBandCorrectionGate on the LL1
            correction head (011g).
        coarse_color_residual (bool): Add a PooledResidualHead from bottleneck
            features for slow tone correction (011i).
        mid_band_residual (bool): Add a PooledResidualHead from 1/4 decoder
            features, band-passed by a 5x5/17x17 smoothing difference (011j).
        extra_lfm_position (str | None): 'bottleneck' or 'decoder' — apply an
            extra LowFreqModBlock on the first min(w2, w4) channels at that
            position (011r position control).

    Optional branches are built inside torch.random.fork_rng so the base
    weights stay bitwise identical across flag combinations (probe arms must
    share the same base initialization).
    """

    def __init__(self, widths=(32, 64, 128, 256), feat=32, use_lfm=True, detail_stem=False,
                 context_dilations=None, low_band_gate=False, coarse_color_residual=False,
                 mid_band_residual=False, extra_lfm_position=None):
        super(WaveletUNet, self).__init__()
        w1, w2, w3, w4 = widths
        self.dwt1 = HaarDWT(3)
        self.iwt1 = HaarIWT(3)

        # encoder（输入为 LL1：3ch，1/2 分辨率）
        self.enc1 = nn.Sequential(nn.Conv2d(3, w1, 3, padding=1), ResBlock(w1), ResBlock(w1))
        self.down1 = WaveletDown(w1, w2)
        self.enc2 = nn.Sequential(ResBlock(w2), ResBlock(w2))
        self.down2 = WaveletDown(w2, w3)
        self.enc3 = nn.Sequential(ResBlock(w3), ResBlock(w3))
        self.down3 = WaveletDown(w3, w4)

        # bottleneck：全局雾幕上下文 + LFM 低频全局调制（011b 新增，针对 DC~0.02 短板）
        self.bottleneck = nn.Sequential(ResBlock(w4), PyramidPool(w4), ResBlock(w4))
        self.lfm = LowFreqModBlock(w4) if use_lfm else None

        # decoder
        self.up3 = WaveletUp(w4, w3)
        self.dec3 = nn.Sequential(nn.Conv2d(w3 * 2, w3, 1), ResBlock(w3), ResBlock(w3))
        self.up2 = WaveletUp(w3, w2)
        self.dec2 = nn.Sequential(nn.Conv2d(w2 * 2, w2, 1), ResBlock(w2), ResBlock(w2))
        self.up1 = WaveletUp(w2, w1)
        self.dec1 = nn.Sequential(nn.Conv2d(w1 * 2, w1, 1), ResBlock(w1), ResBlock(w1))

        # 图像空间小波输出头（在 LL1 的 1/2 分辨率上工作）
        self.ll1_head = nn.Conv2d(w1, 3, 3, padding=1)
        self.gain_net = nn.Sequential(
            nn.Conv2d(w1 + 9 + 9, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 9, 3, padding=1))
        self.free_head = nn.Sequential(
            nn.Conv2d(w1 + 9, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 9, 3, padding=1))
        self.detail_stem = None
        if detail_stem:
            with torch.random.fork_rng(devices=[]):
                self.detail_stem = nn.Conv2d(9, w1, 3, padding=1, bias=False)
            nn.init.zeros_(self.detail_stem.weight)
        self.context_pyramid = None
        if context_dilations:
            with torch.random.fork_rng(devices=[]):
                self.context_pyramid = ContextPyramid(w4, context_dilations)
        self.low_band_gate = None
        if low_band_gate:
            with torch.random.fork_rng(devices=[]):
                self.low_band_gate = LowBandCorrectionGate(w1)
        self.coarse_color_residual = None
        if coarse_color_residual:
            with torch.random.fork_rng(devices=[]):
                self.coarse_color_residual = PooledResidualHead(w4)
        self.mid_band_residual = None
        if mid_band_residual:
            with torch.random.fork_rng(devices=[]):
                self.mid_band_residual = PooledResidualHead(w2)
        self.extra_lfm_position = extra_lfm_position
        self.extra_lfm = None
        # the extra LFM modulates only the leading channels at either position
        # (011r: first 96 channels with widths (64, 96, 160, 256))
        self.extra_lfm_channels = min(w2, w4)
        if extra_lfm_position is not None:
            if extra_lfm_position not in ('bottleneck', 'decoder'):
                raise ValueError(f'Unknown modulation position: {extra_lfm_position}')
            with torch.random.fork_rng(devices=[]):
                self.extra_lfm = LowFreqModBlock(self.extra_lfm_channels)

    def _apply_extra_lfm(self, features, position):
        if self.extra_lfm_position != position:
            return features
        channels = self.extra_lfm_channels
        return torch.cat((self.extra_lfm(features[:, :channels]), features[:, channels:]), dim=1)

    @staticmethod
    def _local_band(residual):
        narrow = F.avg_pool2d(F.pad(residual, (2, 2, 2, 2), mode='replicate'), 5, stride=1)
        broad = F.avg_pool2d(F.pad(residual, (8, 8, 8, 8), mode='replicate'), 17, stride=1)
        return narrow - broad

    def forward(self, x):
        B, _, H, W = x.shape
        ll1, det1 = self._split_subbands(self.dwt1(x))

        if self.detail_stem is None:
            f1 = self.enc1(ll1)
        else:
            f1 = self.enc1[0](ll1) + self.detail_stem(det1)
            f1 = self.enc1[2](self.enc1[1](f1))
        f2 = self.enc2(self.down1(f1))            # w2, 1/4
        f3 = self.enc3(self.down2(f2))            # w3, 1/8
        fb = self.bottleneck(self.down3(f3))      # w4, 1/16
        if self.lfm is not None:
            fb = self.lfm(fb)
        if self.context_pyramid is not None:
            fb = self.context_pyramid(fb)
        fb = self._apply_extra_lfm(fb, 'bottleneck')

        d3 = self.dec3(torch.cat([self._match_size(self.up3(fb), f3), f3], dim=1))
        d2 = self.dec2(torch.cat([self._match_size(self.up2(d3), f2), f2], dim=1))
        d2 = self._apply_extra_lfm(d2, 'decoder')
        d1 = self.dec1(torch.cat([self._match_size(self.up1(d2), f1), f1], dim=1))

        correction = self.ll1_head(d1)
        if self.low_band_gate is not None:
            correction = self.low_band_gate(correction, d1, ll1)
        if self.coarse_color_residual is not None:
            correction = correction + self.coarse_color_residual(fb, ll1)
        if self.mid_band_residual is not None:
            correction = correction + self._local_band(self.mid_band_residual(d2, ll1))
        ll1_rec = ll1 + correction
        strength = F.avg_pool2d(det1.abs(), 3, 1, 1)
        log_gain = self.gain_net(torch.cat([d1, det1, strength], dim=1))
        log_gain = F.avg_pool2d(log_gain, 3, 1, 1)
        gain = torch.exp(log_gain.clamp(-1.0, 2.0))
        free = self.free_head(torch.cat([d1, det1], dim=1))
        det1_rec = det1 * gain + free

        self.aux = (ll1_rec, det1_rec)
        self.last_gain = gain.detach()
        out = self.iwt1(self._pack_subbands(ll1_rec, det1_rec), out_size=(H, W))
        return out

    @staticmethod
    def _split_subbands(x, ch=3):
        B, _, h, w = x.shape
        x = x.view(B, ch, 4, h, w)
        ll = x[:, :, 0]
        det = torch.cat([x[:, :, 1], x[:, :, 2], x[:, :, 3]], dim=1)
        return ll, det

    @staticmethod
    def _pack_subbands(ll, det, ch=3):
        B, _, h, w = ll.shape
        lh, hl, hh = det[:, 0:ch], det[:, ch:2 * ch], det[:, 2 * ch:3 * ch]
        return torch.stack([ll, lh, hl, hh], dim=2).view(B, 4 * ch, h, w)

    @staticmethod
    def _match_size(x, ref):
        # Haar 下采样对奇数尺寸 pad 1（replicate），IWT 输出可能比 skip 大 1；
        # 对齐到 skip 的空间尺寸（多退少补）
        h, w = ref.shape[-2:]
        x = x[:, :, :h, :w]
        if x.shape[-2] < h or x.shape[-1] < w:
            x = F.pad(x, (0, w - x.shape[-1], 0, h - x.shape[-2]), mode='replicate')
        return x
