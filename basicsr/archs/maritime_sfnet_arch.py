"""MaritimeSFNet (014): Three-scale anisotropic spatial-frequency dehazing backbone.

Design specification and mathematical derivations:
    research/014_maritime_sfnet/README.md (Propositions P1-P6 and P1-v2).

Key modules:
    - AnisoRepConv3x3: 4-branch (014a/b/c) or 5-branch with isotropic Center
      Difference (`use_cd=True`, 014d) structural reparameterization
      (plain 3x3 + 1x1 + horizontal diff 1x3 + vertical diff 3x1 [+ center diff 3x3]),
      merged on the fly into a single dense 3x3 conv and losslessly foldable via
      `fold_reparam()` for deployment and frozen-backbone LoRA adaptation.
    - DualChannelSpatialAttention (DCA): DC-preserving multiplicative gating
      in (0, 2) with zero-init identity (1.0 at start) for bottleneck blocks;
      optionally refined by channel-wise grouped 7x7 pixel attention (`channel_spatial=True`).
    - PhysicsAlignedLFM: Low-frequency FiLM modulation parameterized in
      image-space normalized Nyquist units (`cutoff_img`), automatically
      scaled by bottleneck stride `s=4` to prevent scale-bandwidth collapse;
      supports single-grid (014a) or multi-scale 8x8+16x16 grids (014b/c/d).
    - AnisotropicLargeKernelBlock: Parallel 1xK, Kx1, K_sq x K_sq, and 3x3
      depthwise branches targeting maritime horizon/vertical haze anisotropy,
      strictly zero-initialized; deployable at 1/4 bottleneck and 1/2 decoder (`use_l2_aniso_lk`).
    - FrequencyConvexSkipFusion (FCSF): Deep-dominant convex low-band fusion
      plus gated detail injection, optional global scene context (`scene_context=True`),
      and 1x1 channel projection; full-resolution (1x) skip blocks all low-frequency
      haze passthrough (`high_pass_only`).
    - BoundedAffineSoftReconHead (BASR): Strictly positive bounded affine
      reconstruction J_hat = G(K_tilde) * I + B with exact zero-init identity
      (G=1, B=0). Supports original sigmoid gain (014a/b/c) and C^2 zero-curvature
      symmetric-tanh gain + scene contrast gate (`symmetric_gain=True, scene_gate=True`, 014d).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from basicsr.utils.registry import ARCH_REGISTRY


class AnisoRepConv3x3(nn.Module):
    """Four- or five-branch isotropic+anisotropic reparameterized 3x3 convolution.

    Branches during training:
        1. Plain 3x3 convolution
        2. 1x1 pointwise convolution (embedded at kernel center)
        3. Horizontal 1x3 difference convolution ([+w, 0, -w])
        4. Vertical 3x1 difference convolution ([+w; 0; -w])
        5. Optional Center Difference 3x3 convolution (w - sum(w) * delta_center,
           zero DC gain, isotropic radial bandpass targeting 0.02-0.15 mid-band;
           enabled when `use_cd=True`).

    All branches are merged linearly into one dense 3x3 kernel before calling
    F.conv2d, preserving Tensor Core arithmetic intensity and allowing exact
    lossless folding via `fold_reparam()` with zero extra inference cost.
    """

    def __init__(self, dim, use_cd=False):
        super().__init__()
        self.dim = dim
        self.use_cd = bool(use_cd)
        self.plain = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.pw = nn.Conv2d(dim, dim, 1, bias=True)
        self.hd = nn.Conv1d(dim, dim, 3, bias=True)
        self.vd = nn.Conv1d(dim, dim, 3, bias=True)
        self.cd = nn.Conv2d(dim, dim, 3, padding=1, bias=True) if self.use_cd else None
        self.folded_conv = None

        # Variance-balanced initialization so the multi-branch sum has unit gain
        with torch.no_grad():
            if self.use_cd:
                self.plain.weight.mul_(0.45)
                self.pw.weight.mul_(0.45)
                self.hd.weight.mul_(0.30)
                self.vd.weight.mul_(0.30)
                self.cd.weight.mul_(0.25)
                nn.init.zeros_(self.cd.bias)
            else:
                self.plain.weight.mul_(0.5)
                self.pw.weight.mul_(0.5)
                self.hd.weight.mul_(0.35)
                self.vd.weight.mul_(0.35)
            nn.init.zeros_(self.pw.bias)
            nn.init.zeros_(self.hd.bias)
            nn.init.zeros_(self.vd.bias)

    def get_merged_weight(self):
        if self.folded_conv is not None:
            return self.folded_conv.weight, self.folded_conv.bias

        w_plain = self.plain.weight
        w_pw = F.pad(self.pw.weight, (1, 1, 1, 1))
        zero_hd = torch.zeros_like(self.hd.weight)
        w_hd = torch.stack((self.hd.weight, zero_hd, -self.hd.weight), dim=3)
        zero_vd = torch.zeros_like(self.vd.weight)
        w_vd = torch.stack((self.vd.weight, zero_vd, -self.vd.weight), dim=2)
        w = w_plain + w_pw + w_hd + w_vd
        b = self.plain.bias + self.pw.bias + self.hd.bias + self.vd.bias
        if self.cd is not None:
            w_cd_sum = self.cd.weight.sum(dim=(2, 3), keepdim=True)
            w_cd = self.cd.weight - F.pad(w_cd_sum, (1, 1, 1, 1))
            w = w + w_cd
            b = b + self.cd.bias
        return w, b

    @torch.no_grad()
    def fold_reparam(self):
        """Losslessly fold all training branches into one plain 3x3 conv."""
        if self.folded_conv is not None:
            return
        w, b = self.get_merged_weight()
        conv = nn.Conv2d(self.dim, self.dim, 3, padding=1, bias=True).to(
            device=w.device, dtype=w.dtype)
        conv.weight.copy_(w)
        conv.bias.copy_(b)
        self.folded_conv = conv
        self.plain = None
        self.pw = None
        self.hd = None
        self.vd = None
        self.cd = None

    def forward(self, x):
        if self.folded_conv is not None:
            return self.folded_conv(x)
        w, b = self.get_merged_weight()
        return F.conv2d(x, w, b, stride=1, padding=1)


class DualChannelSpatialAttention(nn.Module):
    """DC-preserving channel (GAP) + 7x7 spatial multiplicative gating.

    Output gate lies strictly in (0, 2) and starts at exact identity 1.0 via
    zero initialization, avoiding early activation attenuation.
    When `channel_spatial=True`, adds a zero-initialized channel-wise grouped
    7x7 residual refinement (`pa`) over interleaved `(x_c, p1_c)` pairs so each
    channel obtains its own spatial modulation without gradient deadlock.
    """

    def __init__(self, dim, reduction=4, channel_spatial=False):
        super().__init__()
        hidden = max(dim // reduction, 16)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1))
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='replicate')
        nn.init.zeros_(self.ca[-1].weight)
        nn.init.zeros_(self.ca[-1].bias)
        nn.init.zeros_(self.sa.weight)
        nn.init.zeros_(self.sa.bias)
        if channel_spatial:
            self.pa = nn.Conv2d(
                2 * dim, dim, 7, padding=3, padding_mode='replicate', groups=dim)
            nn.init.zeros_(self.pa.weight)
            nn.init.zeros_(self.pa.bias)
        else:
            self.pa = None

    def forward(self, x):
        c_attn = self.ca(x)
        s_in = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        s_attn = self.sa(s_in)
        p1 = c_attn + s_attn
        if self.pa is not None:
            b, c, h, w = x.shape
            pa_in = torch.stack((x, p1), dim=2).reshape(b, 2 * c, h, w)
            logits = p1 + self.pa(pa_in)
        else:
            logits = p1
        return x * (2.0 * torch.sigmoid(logits))


class MaritimeBlock(nn.Module):
    """Dense 3x3 residual block with optional reparameterization and DCA."""

    def __init__(self, dim, use_attn=False, attn_reduction=4, use_reparam=True,
                 use_cd=False, channel_spatial_attn=False):
        super().__init__()
        self.conv1 = (
            AnisoRepConv3x3(dim, use_cd=use_cd)
            if use_reparam else nn.Conv2d(dim, dim, 3, padding=1))
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        with torch.no_grad():
            self.conv2.weight.mul_(0.5)
        self.attn = (
            DualChannelSpatialAttention(
                dim, reduction=attn_reduction, channel_spatial=channel_spatial_attn)
            if use_attn else None)

    def forward(self, x):
        res = self.act(self.conv1(x)) + x
        res = self.conv2(res)
        if self.attn is not None:
            res = self.attn(res)
        return res + x


class PhysicsAlignedLFM(nn.Module):
    """Scale-bandwidth-aligned low-frequency FiLM modulation (Propositions P2 & P3).

    The low-pass cutoff is specified in *image-space* normalized Nyquist units
    (`cutoff_img`) and automatically scaled by `scale_factor` (4 at 1/4 scale)
    so the physical frequency band never collapses when bottleneck depth changes.
    Supports single-grid (014a) or multi-scale pooled grids (014b/c).
    No normalization; strictly zero-initialized to exact identity.
    """

    def __init__(self, dim, cutoff_img=0.020, edge_img=0.010, scale_factor=4,
                 mod_grids=(16,), reduction=4):
        super().__init__()
        self.cutoff_feat = min(float(cutoff_img) * scale_factor, 0.5)
        self.edge_feat = min(float(edge_img) * scale_factor, 0.25)
        self.mod_grids = tuple(mod_grids)
        hidden = max(dim // reduction, 32)
        self.mod = nn.Sequential(
            nn.Conv2d(dim * len(self.mod_grids), hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim * 2, 1))
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    @staticmethod
    def _soft_lowpass_mask(h, w, cutoff, edge, device):
        nyq = max(min(h, w) / 2.0, 1.0)
        v = torch.fft.fftfreq(h, device=device) * h
        u = torch.fft.rfftfreq(w, device=device) * w
        r = torch.sqrt(v[:, None] ** 2 + u[None, :] ** 2) / nyq
        return torch.sigmoid((cutoff - r) / edge)[None, None, :, :]

    def forward(self, x):
        _, _, h, w = x.shape
        mask = self._soft_lowpass_mask(h, w, self.cutoff_feat, self.edge_feat, x.device)
        spec = torch.fft.rfft2(x.float(), norm='ortho')
        low = torch.fft.irfft2(spec * mask, s=(h, w), norm='ortho').to(x.dtype)

        gmax = max(self.mod_grids)
        target_size = (min(h, gmax), min(w, gmax))
        pooled = []
        for grid in self.mod_grids:
            p = F.adaptive_avg_pool2d(x, (min(h, grid), min(w, grid)))
            if p.shape[-2:] != target_size:
                p = F.interpolate(p, size=target_size, mode='bilinear', align_corners=False)
            pooled.append(p)
        mod = self.mod(torch.cat(pooled, dim=1))
        mod = F.interpolate(mod, size=(h, w), mode='bilinear', align_corners=False)
        gamma, beta = mod.chunk(2, dim=1)
        return x + gamma * low + beta


class AnisotropicLargeKernelBlock(nn.Module):
    """Maritime anisotropic multi-shape large-kernel spatial branch (Proposition P3).

    Splits channels into four equal groups processed in parallel by:
        - 1 x k_strip horizontal depthwise conv (horizon / iso-depth sea bands)
        - k_strip x 1 vertical depthwise conv (sky -> horizon -> near-sea haze gradient)
        - k_square x k_square square depthwise conv (patchy non-uniform haze)
        - 3 x 3 local depthwise conv (object boundary continuity)

    Strictly zero-initialized output projection ensures exact identity at start.
    """

    def __init__(self, dim, k_strip=31, k_square=15):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(f'AnisotropicLargeKernelBlock requires dim % 4 == 0, got {dim}')
        q = dim // 4
        self.in_proj = nn.Conv2d(dim, dim, 1)
        self.dw_h = nn.Conv2d(
            q, q, (1, k_strip), padding=(0, k_strip // 2),
            padding_mode='replicate', groups=q)
        self.dw_v = nn.Conv2d(
            q, q, (k_strip, 1), padding=(k_strip // 2, 0),
            padding_mode='replicate', groups=q)
        self.dw_sq = nn.Conv2d(
            q, q, k_square, padding=k_square // 2,
            padding_mode='replicate', groups=q)
        self.dw_loc = nn.Conv2d(
            q, q, 3, padding=1, padding_mode='replicate', groups=q)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        x0, x1, x2, x3 = self.in_proj(x).chunk(4, dim=1)
        mixed = torch.cat((
            self.dw_h(x0),
            self.dw_v(x1),
            self.dw_sq(x2),
            self.dw_loc(x3),
        ), dim=1)
        return x + self.out_proj(F.gelu(mixed))


class AnisoSpatialFreqBlock(nn.Module):
    """Dual-domain bottleneck block pairing PhysicsAlignedLFM with optional AnisoLK."""

    def __init__(self, dim, cutoff_img=0.020, edge_img=0.010, scale_factor=4,
                 mod_grids=(16,), use_aniso_lk=False, lk_strip=31, lk_square=15):
        super().__init__()
        self.ms_lfm = PhysicsAlignedLFM(
            dim,
            cutoff_img=cutoff_img,
            edge_img=edge_img,
            scale_factor=scale_factor,
            mod_grids=mod_grids)
        self.aniso_lk = (
            AnisotropicLargeKernelBlock(dim, k_strip=lk_strip, k_square=lk_square)
            if use_aniso_lk else None)

    def forward(self, x):
        x = self.ms_lfm(x)
        if self.aniso_lk is not None:
            x = self.aniso_lk(x)
        return x


class FrequencyConvexSkipFusion(nn.Module):
    """Deep-dominant frequency-decoupled convex skip fusion (Proposition P4).

    When `high_pass_only=False` (1/4 & 1/2 scales):
        fused = out_proj(deep + alpha * (skip_low - deep_low) + beta * skip_detail)
        with alpha in (0, 1) initialized via `alpha_init_bias` (default -2.0,
        alpha_0 ~ 0.12; -1.2 gives ~0.23) so dehazed deep features dominate the
        low-frequency base.
    When `high_pass_only=True` (1x full resolution):
        alpha === 0; blocks shallow hazy low frequencies completely and injects
        only gated high-pass structural detail.
    When `scene_context=True` (014d):
        adds a zero-initialized global GAP scene branch to the gate logits so
        light-haze scenes can smoothly relax low-band suppression while heavy
        haze keeps alpha near zero.
    """

    def __init__(self, dim, reduction=4, high_pass_only=False, lp_stride=4,
                 scene_context=False, alpha_init_bias=-2.0):
        super().__init__()
        self.dim = dim
        self.high_pass_only = high_pass_only
        self.lp_stride = lp_stride
        hidden = max(dim // reduction, 16)
        in_ch = 2 * dim if high_pass_only else 3 * dim
        out_ch = dim if high_pass_only else 2 * dim
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1))
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            if high_pass_only:
                self.gate[-1].bias.fill_(-1.0)  # beta_0 ~ 0.27
            else:
                self.gate[-1].bias[:dim].fill_(float(alpha_init_bias))
                self.gate[-1].bias[dim:].fill_(0.0)  # beta_0 = 0.50
        if scene_context:
            self.scene_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, hidden, 1),
                nn.GELU(),
                nn.Conv2d(hidden, out_ch, 1))
            nn.init.zeros_(self.scene_gate[-1].weight)
            nn.init.zeros_(self.scene_gate[-1].bias)
        else:
            self.scene_gate = None
        self.out_proj = nn.Conv2d(dim, dim, 1)
        self.last_alpha_mean = None
        self.last_beta_mean = None

    @staticmethod
    def _low_pass(x, stride):
        h, w = x.shape[-2:]
        lh = max(1, (h + stride - 1) // stride)
        lw = max(1, (w + stride - 1) // stride)
        pooled = F.adaptive_avg_pool2d(x, (lh, lw))
        return F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)

    @staticmethod
    def _pool2(x):
        h, w = x.shape[-2:]
        return F.adaptive_avg_pool2d(x, (max(1, (h + 1) // 2), max(1, (w + 1) // 2)))

    def forward(self, deep, skip):
        skip_low = self._low_pass(skip, self.lp_stride)
        skip_detail = skip - skip_low

        if self.high_pass_only:
            evidence = torch.cat((self._pool2(deep), self._pool2(skip_detail.abs())), dim=1)
            logits = F.interpolate(
                self.gate(evidence), size=deep.shape[-2:], mode='bilinear', align_corners=False)
            if self.scene_gate is not None:
                logits = logits + self.scene_gate(evidence)
            beta = torch.sigmoid(logits)
            self.last_alpha_mean = None
            self.last_beta_mean = beta.detach().mean()
            return self.out_proj(deep + beta * skip_detail)

        deep_low = self._low_pass(deep, self.lp_stride)
        evidence = torch.cat((
            self._pool2(deep),
            self._pool2(skip_low - deep_low),
            self._pool2(skip_detail.abs()),
        ), dim=1)
        logits = F.interpolate(
            self.gate(evidence), size=deep.shape[-2:], mode='bilinear', align_corners=False)
        if self.scene_gate is not None:
            logits = logits + self.scene_gate(evidence)
        alpha, beta = torch.sigmoid(logits).chunk(2, dim=1)
        self.last_alpha_mean = alpha.detach().mean()
        self.last_beta_mean = beta.detach().mean()
        fused = deep + alpha * (skip_low - deep_low) + beta * skip_detail
        return self.out_proj(fused)


class BoundedAffineSoftReconHead(nn.Module):
    """Strictly positive bounded affine soft reconstruction head (Propositions P1 & P1-v2).

        J_hat = G(K_tilde) * I + B

    Two bounded positive parameterizations with exact zero-init identity (G=1, B=0):
        1. `symmetric_gain=False` (014a/b/c):
               G(k) = g_min + (g_max - g_min) * sigmoid(k + b0),  G''(0) > 0.
        2. `symmetric_gain=True` (014d, Proposition P1-v2):
               G(k) = 1 + a_+ * tanh(k / a_+)  for k >= 0
                      1 + a_- * tanh(k / a_-)  for k < 0
           with a_+ = g_max - 1, a_- = 1 - g_min. Satisfies G(0)=1, G'(0)=1,
           G''(0)=0 (C^2 zero-curvature at identity), eliminating Jensen's
           convexity overshoot bias on light-haze scenes while keeping G in (g_min, g_max).

    When `scene_gate=True` (014d), a lightweight global scene contrast MLP over
    [GAP(feat), GAP(image), RMS_contrast(image)] scales (raw_k, bias) in (0, 2)
    with exact 1.0 initialization so light-haze scenes can damp over-correction.
    """

    def __init__(self, dim, out_ch=3, g_min=0.5, g_max=5.0,
                 symmetric_gain=False, scene_gate=False):
        super().__init__()
        if not (0.0 < g_min < 1.0 < g_max):
            raise ValueError(f'Require 0 < g_min < 1 < g_max, got ({g_min}, {g_max})')
        self.g_min = float(g_min)
        self.g_max = float(g_max)
        self.a_pos = self.g_max - 1.0
        self.a_neg = 1.0 - self.g_min
        self.b0 = -math.log(self.a_pos / self.a_neg)
        self.symmetric_gain = bool(symmetric_gain)
        self.head = nn.Conv2d(dim, out_ch * 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if scene_gate:
            scene_in = dim + 2 * out_ch
            scene_hidden = max(dim // 2, 16)
            self.scene_mlp = nn.Sequential(
                nn.Conv2d(scene_in, scene_hidden, 1),
                nn.GELU(),
                nn.Conv2d(scene_hidden, out_ch * 2, 1))
            nn.init.zeros_(self.scene_mlp[-1].weight)
            nn.init.zeros_(self.scene_mlp[-1].bias)
        else:
            self.scene_mlp = None
        self.last_gain_mean = None
        self.last_bias_abs = None
        self.last_scene_scale = None

    def forward(self, feat, image):
        raw_k, bias = self.head(feat).chunk(2, dim=1)
        if self.scene_mlp is not None:
            g_feat = F.adaptive_avg_pool2d(feat, 1)
            g_img_mean = F.adaptive_avg_pool2d(image, 1)
            g_img_std = torch.sqrt(
                image.var(dim=(-2, -1), keepdim=True, unbiased=False) + 1e-6)
            scene_scale = 2.0 * torch.sigmoid(
                self.scene_mlp(torch.cat((g_feat, g_img_mean, g_img_std), dim=1)))
            s_k, s_b = scene_scale.chunk(2, dim=1)
            raw_k = raw_k * s_k
            bias = bias * s_b
            self.last_scene_scale = scene_scale.detach().mean()
        else:
            self.last_scene_scale = None

        if self.symmetric_gain:
            gain = torch.where(
                raw_k >= 0,
                1.0 + self.a_pos * torch.tanh(raw_k / self.a_pos),
                1.0 + self.a_neg * torch.tanh(raw_k / self.a_neg))
        else:
            gain = self.g_min + (self.g_max - self.g_min) * torch.sigmoid(raw_k + self.b0)
        self.last_gain_mean = gain.detach().mean()
        self.last_bias_abs = bias.detach().abs().mean()
        return gain * image + bias


class BilinearConvUp(nn.Module):
    """Smooth bilinear 2x upsampling followed by 3x3 conv (avoids 2x2 grid artifacts)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x, target_size=None):
        if target_size is None:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.conv(x)


@ARCH_REGISTRY.register()
class MaritimeSFNet(nn.Module):
    """Three-scale physics-aligned anisotropic spatial-frequency dehazing backbone.

    Args:
        in_ch (int): Input RGB channels. Default: 3.
        out_ch (int): Output RGB channels. Default: 3.
        widths (tuple[int, int, int]): Channels at 1x, 1/2, 1/4 scales.
            Default: (36, 72, 144).
        depths (tuple[int, int, int]): Block counts at L1 (enc & dec each),
            L2 (enc & dec each), and L3 bottleneck. Default: (3, 3, 8).
        use_reparam (bool): Use AnisoRepConv3x3 in MaritimeBlock. Default: True.
        use_cd (bool): Enable 5th isotropic Center-Difference branch inside
            AnisoRepConv3x3 (zero DC gain, targets 0.02-0.15 mid-band, folds
            losslessly at zero inference cost; 014d). Default: False.
        channel_spatial_attn (bool): Add channel-wise grouped 7x7 pixel
            attention refinement to bottleneck DCA (014d). Default: False.
        use_aniso_lk (bool): Enable AnisotropicLargeKernelBlock in bottleneck
            ASFB (014b/c/d). Default: False.
        lfm_cutoff_img (float): Low-pass cutoff in *image-space* normalized
            Nyquist units (automatically scaled x4 at 1/4 bottleneck).
            Default: 0.020 (014a); use 0.025 for 014b/c/d.
        lfm_edge_img (float): Soft transition width in image-space Nyquist units.
            Default: 0.010.
        lfm_mod_grids (tuple[int, ...]): Modulation pooling grid sizes.
            Default: (16,) for 014a; use (8, 16) for 014b/c/d.
        lk_strip (int): Strip kernel length for 1xK / Kx1 branches. Default: 31.
        lk_square (int): Square large kernel size. Default: 15.
        fcsf_scene_context (bool): Add global scene GAP branch to FCSF gates
            (014d). Default: False.
        fcsf_alpha_bias (float): Initial logit bias for FCSF low-band gate alpha.
            Default: -2.0 (014a/b/c); use -1.2 for 014d.
        basr_symmetric (bool): Use C^2 zero-curvature asymmetric-tanh gain in
            BASR head (Proposition P1-v2, 014d). Default: False.
        basr_scene_gate (bool): Add global scene contrast modulator in BASR head
            (014d). Default: False.
        basr_g_max (float): Upper bound of BASR multiplicative gain. Default: 5.0
            (014a/b/c); use 4.0 for 014d.
        use_l2_aniso_lk (bool): Add lightweight mid-scale AnisotropicLargeKernelBlock
            after dec2 (1/2 scale, 256x256) targeting 0.02-0.15 mid-band (014d).
            Default: False.
        l2_lk_strip (int): Strip kernel size for 1/2-scale AnisoLK. Default: 15.
        l2_lk_square (int): Square kernel size for 1/2-scale AnisoLK. Default: 7.
        use_mimo_input (bool): Inject downsampled hazy RGB at 1/2 and 1/4
            encoder stages (014c/d). Default: False.
        use_deep_supervision (bool): Enable 1/4 and 1/2 training-only auxiliary
            BASR heads (014c, skipped at eval for zero inference overhead).
            Default: False.
        activation_checkpointing (bool): Gradient checkpointing for heavy VRAM
            configs. Default: False (enable on 24G GPUs with bs=16).
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        widths=(36, 72, 144),
        depths=(3, 3, 8),
        use_reparam=True,
        use_cd=False,
        channel_spatial_attn=False,
        use_aniso_lk=False,
        lfm_cutoff_img=0.020,
        lfm_edge_img=0.010,
        lfm_mod_grids=(16,),
        lk_strip=31,
        lk_square=15,
        fcsf_scene_context=False,
        fcsf_alpha_bias=-2.0,
        basr_symmetric=False,
        basr_scene_gate=False,
        basr_g_max=5.0,
        use_l2_aniso_lk=False,
        l2_lk_strip=15,
        l2_lk_square=7,
        use_mimo_input=False,
        use_deep_supervision=False,
        activation_checkpointing=False,
    ):
        super().__init__()
        if len(widths) != 3 or len(depths) != 3:
            raise ValueError(f'MaritimeSFNet is a 3-scale architecture; got widths={widths}, depths={depths}')
        w1, w2, w3 = widths
        d1, d2, d3 = depths
        if d3 < 3:
            raise ValueError(f'Bottleneck depth d3 must be >= 3 to place 2 ASFB blocks, got {d3}')

        self.activation_checkpointing = activation_checkpointing
        self.use_mimo_input = use_mimo_input
        self.use_deep_supervision = use_deep_supervision

        # Encoder
        self.stem = nn.Conv2d(in_ch, w1, 3, padding=1)
        self.enc1 = nn.Sequential(*(
            MaritimeBlock(w1, use_attn=False, use_reparam=use_reparam, use_cd=use_cd)
            for _ in range(d1)))
        self.down1 = nn.Sequential(nn.Conv2d(w1, w2, 3, stride=2, padding=1), nn.GELU())

        self.enc2 = nn.Sequential(*(
            MaritimeBlock(w2, use_attn=False, use_reparam=use_reparam, use_cd=use_cd)
            for _ in range(d2)))
        self.down2 = nn.Sequential(nn.Conv2d(w2, w3, 3, stride=2, padding=1), nn.GELU())

        # Optional MIMO multi-scale hazy input injections (zero-init for smooth start)
        if use_mimo_input:
            self.mimo_in2 = nn.Conv2d(in_ch, w2, 3, padding=1)
            self.mimo_in3 = nn.Conv2d(in_ch, w3, 3, padding=1)
            nn.init.zeros_(self.mimo_in2.weight)
            nn.init.zeros_(self.mimo_in2.bias)
            nn.init.zeros_(self.mimo_in3.weight)
            nn.init.zeros_(self.mimo_in3.bias)
        else:
            self.mimo_in2 = None
            self.mimo_in3 = None

        # 1/4 Bottleneck: split d3 blocks into 3 segments separated by 2 ASFB blocks
        # For d3=8: 2 + ASFB1 + 3 + ASFB2 + 3 (matches DEA-Net [2, 5] spacing)
        b_seg1 = d3 // 4
        b_seg2 = (d3 - b_seg1) // 2
        b_seg3 = d3 - b_seg1 - b_seg2
        self.bot_seg1 = nn.Sequential(*(
            MaritimeBlock(
                w3, use_attn=True, use_reparam=use_reparam,
                use_cd=use_cd, channel_spatial_attn=channel_spatial_attn)
            for _ in range(b_seg1)))
        self.asfb1 = AnisoSpatialFreqBlock(
            w3, cutoff_img=lfm_cutoff_img, edge_img=lfm_edge_img, scale_factor=4,
            mod_grids=lfm_mod_grids, use_aniso_lk=use_aniso_lk,
            lk_strip=lk_strip, lk_square=lk_square)
        self.bot_seg2 = nn.Sequential(*(
            MaritimeBlock(
                w3, use_attn=True, use_reparam=use_reparam,
                use_cd=use_cd, channel_spatial_attn=channel_spatial_attn)
            for _ in range(b_seg2)))
        self.asfb2 = AnisoSpatialFreqBlock(
            w3, cutoff_img=lfm_cutoff_img, edge_img=lfm_edge_img, scale_factor=4,
            mod_grids=lfm_mod_grids, use_aniso_lk=use_aniso_lk,
            lk_strip=lk_strip, lk_square=lk_square)
        self.bot_seg3 = nn.Sequential(*(
            MaritimeBlock(
                w3, use_attn=True, use_reparam=use_reparam,
                use_cd=use_cd, channel_spatial_attn=channel_spatial_attn)
            for _ in range(b_seg3)))

        # Decoder + frequency-decoupled convex skip fusion
        self.fcsf3 = FrequencyConvexSkipFusion(
            w3, high_pass_only=False,
            scene_context=fcsf_scene_context, alpha_init_bias=fcsf_alpha_bias)
        self.up2 = BilinearConvUp(w3, w2)
        self.fcsf2 = FrequencyConvexSkipFusion(
            w2, high_pass_only=False,
            scene_context=fcsf_scene_context, alpha_init_bias=fcsf_alpha_bias)
        self.dec2 = nn.Sequential(*(
            MaritimeBlock(w2, use_attn=False, use_reparam=use_reparam, use_cd=use_cd)
            for _ in range(d2)))
        self.dec2_aniso_lk = (
            AnisotropicLargeKernelBlock(w2, k_strip=l2_lk_strip, k_square=l2_lk_square)
            if use_l2_aniso_lk else None)

        self.up1 = BilinearConvUp(w2, w1)
        self.fcsf1 = FrequencyConvexSkipFusion(
            w1, high_pass_only=True, scene_context=fcsf_scene_context)
        self.dec1 = nn.Sequential(*(
            MaritimeBlock(w1, use_attn=False, use_reparam=use_reparam, use_cd=use_cd)
            for _ in range(d1)))

        # Output head: strictly positive bounded affine soft reconstruction
        self.basr_head = BoundedAffineSoftReconHead(
            w1, out_ch=out_ch, g_max=basr_g_max,
            symmetric_gain=basr_symmetric, scene_gate=basr_scene_gate)

        # Training-only auxiliary heads for multi-scale deep supervision (014c)
        if use_deep_supervision:
            self.aux_head_l3 = BoundedAffineSoftReconHead(
                w3, out_ch=out_ch, g_max=basr_g_max,
                symmetric_gain=basr_symmetric, scene_gate=basr_scene_gate)
            self.aux_head_l2 = BoundedAffineSoftReconHead(
                w2, out_ch=out_ch, g_max=basr_g_max,
                symmetric_gain=basr_symmetric, scene_gate=basr_scene_gate)
        else:
            self.aux_head_l3 = None
            self.aux_head_l2 = None

        self.aux_outputs = {}
        self.diagnostics = {}

    def _run(self, module, *inputs):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    @torch.no_grad()
    def fold_reparam(self):
        """Fold all AnisoRepConv3x3 branches into plain 3x3 convolutions."""
        for module in self.modules():
            if isinstance(module, AnisoRepConv3x3):
                module.fold_reparam()

    def iter_lora_target_modules(self):
        """Return physics-decoupled domain-sensitive module groups for downstream LoRA.

        Intended for real-maritime adaptation after freezing the folded local
        texture backbone (see research/014_maritime_sfnet/README.md §4).
        """
        targets = {
            'ms_lfm': [self.asfb1.ms_lfm.mod, self.asfb2.ms_lfm.mod],
            'aniso_lk': [
                m for m in (self.asfb1.aniso_lk, self.asfb2.aniso_lk, self.dec2_aniso_lk)
                if m is not None
            ],
            'fcsf': [self.fcsf3, self.fcsf2, self.fcsf1],
            'basr': [self.basr_head],
        }
        return targets

    def forward(self, image):
        h, w = image.shape[-2:]
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate') if (pad_h or pad_w) else image

        need_img_1_2 = self.use_mimo_input or (self.use_deep_supervision and self.training)
        need_img_1_4 = need_img_1_2
        img_1_2 = (
            F.interpolate(padded, scale_factor=0.5, mode='bilinear', align_corners=False)
            if need_img_1_2 else None)
        img_1_4 = (
            F.interpolate(padded, scale_factor=0.25, mode='bilinear', align_corners=False)
            if need_img_1_4 else None)

        # Encoder
        s1 = self._run(self.enc1, self.stem(padded))
        x2 = self.down1(s1)
        if self.mimo_in2 is not None:
            x2 = x2 + self.mimo_in2(img_1_2)
        s2 = self._run(self.enc2, x2)

        x3 = self.down2(s2)
        if self.mimo_in3 is not None:
            x3 = x3 + self.mimo_in3(img_1_4)

        # 1/4 Bottleneck
        b = self._run(self.bot_seg1, x3)
        b = self._run(self.asfb1, b)
        b = self._run(self.bot_seg2, b)
        b = self._run(self.asfb2, b)
        b = self._run(self.bot_seg3, b)
        b = self._run(self.fcsf3, b, x3)

        # Decoder
        d2 = self._run(self.fcsf2, self.up2(b, target_size=s2.shape[-2:]), s2)
        d2 = self._run(self.dec2, d2)
        if self.dec2_aniso_lk is not None:
            d2 = self._run(self.dec2_aniso_lk, d2)

        d1 = self._run(self.fcsf1, self.up1(d2, target_size=s1.shape[-2:]), s1)
        d1 = self._run(self.dec1, d1)

        out = self.basr_head(d1, padded)

        # Training-only multi-scale deep supervision outputs (zero eval cost)
        if self.use_deep_supervision and self.training:
            self.aux_outputs = {
                '1/4': self.aux_head_l3(b, img_1_4),
                '1/2': self.aux_head_l2(d2, img_1_2),
            }
        else:
            self.aux_outputs = {}

        self.diagnostics = {
            'diag_basr_gain': self.basr_head.last_gain_mean,
            'diag_basr_bias': self.basr_head.last_bias_abs,
            'diag_basr_scene': self.basr_head.last_scene_scale,
            'diag_fcsf3_alpha': self.fcsf3.last_alpha_mean,
            'diag_fcsf2_alpha': self.fcsf2.last_alpha_mean,
            'diag_fcsf1_beta': self.fcsf1.last_beta_mean,
        }

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out
