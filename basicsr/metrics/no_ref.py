"""No-reference image quality metrics for dehazing evaluation.

All metrics follow the framework convention: they take the dehazed image
(``img``, HWC uint8 RGB ndarray, as produced by ``tensor2img`` in validation)
and return a scalar; the optional ``img2`` argument (GT) is ignored.

- FADE (Fog Aware Density Evaluator, TIP 2015): fog-density specific,
  lower = less fog = better dehazing. Requires the ``fade-python`` package.
- NIQE / BRISQUE (pyiqa implementation): natural-scene-statistic quality,
  lower = better. Requires the ``pyiqa`` package (model weights are
  downloaded automatically on first use).
- MUSIQ / MANIQA / CLIP-IQA (pyiqa implementation): deep NR-IQA metrics
  widely reported in recent restoration papers, higher = better. Deep
  models run on GPU when available (much faster than CPU).

Note: these metrics are much slower than PSNR/SSIM (tens of ms per image).
Recommended usage is final evaluation / real-world data (test.py or the
standalone script scripts/metrics/calculate_nr_metrics.py), not per-1k-iter
training validation.
"""

import numpy as np

from basicsr.utils.registry import METRIC_REGISTRY

_IQA_MODEL_CACHE = {}


def _get_iqa_metric(name, device=None):
    """Lazily create (and cache) a pyiqa metric to avoid re-init per image."""
    if device is None:
        device = 'cpu'
    key = (name, device)
    if key not in _IQA_MODEL_CACHE:
        try:
            import pyiqa
        except ImportError:
            raise ImportError('pyiqa is required for NR-IQA metrics. Install it with: pip install pyiqa')
        _IQA_MODEL_CACHE[key] = pyiqa.create_metric(name, device=device)
    return _IQA_MODEL_CACHE[key]


def _to_iqa_input(img, device='cpu'):
    """Convert HWC uint8 RGB ndarray to the CHW float tensor that pyiqa expects."""
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def _deep_iqa(name, img):
    """Run a deep NR-IQA metric on GPU when available."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = _get_iqa_metric(name, device)
    score = metric(_to_iqa_input(img, device))
    return float(score.item() if hasattr(score, 'item') else score)


@METRIC_REGISTRY.register()
def calculate_fade(img, img2=None, **kwargs):
    """FADE fog density metric (lower = less fog = better).

    Args:
        img (ndarray): Dehazed image, HWC uint8 RGB.
        img2 (ndarray): Unused (kept for framework signature compatibility).

    Returns:
        float: FADE score.
    """
    try:
        from pyfade import fade_array
    except ImportError:
        raise ImportError('fade-python is required for FADE. Install it with: pip install fade-python')

    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return float(fade_array(img))


@METRIC_REGISTRY.register()
def calculate_niqe(img, img2=None, **kwargs):
    """NIQE natural image quality evaluator (lower = better).

    Args:
        img (ndarray): Dehazed image, HWC uint8 RGB.
        img2 (ndarray): Unused.

    Returns:
        float: NIQE score.
    """
    metric = _get_iqa_metric('niqe')
    score = metric(_to_iqa_input(img))
    return float(score.item() if hasattr(score, 'item') else score)


@METRIC_REGISTRY.register()
def calculate_brisque(img, img2=None, **kwargs):
    """BRISQUE blind image quality metric (lower = better).

    Args:
        img (ndarray): Dehazed image, HWC uint8 RGB.
        img2 (ndarray): Unused.

    Returns:
        float: BRISQUE score.
    """
    metric = _get_iqa_metric('brisque')
    score = metric(_to_iqa_input(img))
    return float(score.item() if hasattr(score, 'item') else score)


@METRIC_REGISTRY.register()
def calculate_musiq(img, img2=None, **kwargs):
    """MUSIQ multi-scale NR-IQA (ICCV 2021, higher = better).

    Args:
        img (ndarray): Dehazed image, HWC uint8 RGB.
        img2 (ndarray): Unused.

    Returns:
        float: MUSIQ score.
    """
    return _deep_iqa('musiq', img)


@METRIC_REGISTRY.register()
def calculate_maniqa(img, img2=None, **kwargs):
    """MANIQA NR-IQA (CVPR 2022, NTIRE official metric, higher = better).

    Args:
        img (ndarray): Dehazed image, HWC uint8 RGB.
        img2 (ndarray): Unused.

    Returns:
        float: MANIQA score.
    """
    return _deep_iqa('maniqa', img)


@METRIC_REGISTRY.register()
def calculate_clipiqa(img, img2=None, **kwargs):
    """CLIP-IQA semantic NR-IQA (AAAI 2023, higher = better).

    Args:
        img (ndarray): Dehazed image, HWC uint8 RGB.
        img2 (ndarray): Unused.

    Returns:
        float: CLIP-IQA score.
    """
    return _deep_iqa('clipiqa', img)
