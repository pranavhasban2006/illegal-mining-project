"""Bi-temporal change detection: find ground that became mining between two dates.

The steps are:
1. Normalize each scene on its own, so that a change in season or haze does not
   look like a change on the ground.
2. Compare the two dates to get delta features.
3. Keep pixels that look like mining at t1 but not at t0.
4. Clean up the mask (remove speckle, fill small holes) and drop tiny patches.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .indices import FEATURE_NAMES


def normalize_per_scene(stack: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Z-score every feature using only the statistics of this scene.

    Each scene gets its own mean and standard deviation per feature. Invalid pixels
    (cloud, nodata) are excluded from those statistics and come out as 0.

    Args:
        stack: (H, W, F) feature array.
        valid: optional (H, W) boolean mask; True means the pixel is usable.

    Returns:
        (H, W, F) float32 array with roughly zero mean and unit variance per feature.
    """
    stack = np.asarray(stack, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(stack).all(axis=-1)
    out = np.zeros_like(stack)
    sel = stack[valid]  # (N, F)
    if sel.size == 0:
        return out
    mean = sel.mean(axis=0)
    std = sel.std(axis=0)
    std[std < 1e-9] = 1.0
    out = (stack - mean) / std
    out[~valid] = 0.0
    return out.astype(np.float32)


def delta_features(stack0: np.ndarray, stack1: np.ndarray) -> np.ndarray:
    """Return the difference t1 - t0 for every feature, shape (H, W, F)."""
    if stack0.shape != stack1.shape:
        raise ValueError(
            f"Scenes must be on the same grid: {stack0.shape} vs {stack1.shape}"
        )
    return (stack1 - stack0).astype(np.float32)


def spectral_disturbance_score(stack0: np.ndarray, stack1: np.ndarray) -> np.ndarray:
    """A simple, rule-based mining score in [0, 1] (no trained model needed).

    Mining turns green forest into bare soil and sometimes water. The score is
    high where NDVI fell and bare-soil index (BSI) rose between the two dates.
    It is used by the offline demo and as a baseline to compare the trained model
    against.
    """
    names = list(FEATURE_NAMES)
    ndvi0, ndvi1 = stack0[..., names.index("NDVI")], stack1[..., names.index("NDVI")]
    bsi0, bsi1 = stack0[..., names.index("BSI")], stack1[..., names.index("BSI")]
    drop_ndvi = np.clip((ndvi0 - ndvi1) / 0.3, 0.0, 1.0)
    rise_bsi = np.clip((bsi1 - bsi0) / 0.2, 0.0, 1.0)
    return (drop_ndvi * rise_bsi).astype(np.float32)


def rule_mining_probability(stack: np.ndarray) -> np.ndarray:
    """A rule-based 'looks like open mining' score in [0, 1] for one date.

    Open mining pits and spoil are bare (low NDVI) and soil-like (high BSI). Water
    and forest both score low. Used by the offline demo, where no trained model exists.
    """
    names = list(FEATURE_NAMES)
    ndvi_ = stack[..., names.index("NDVI")]
    bsi_ = stack[..., names.index("BSI")]
    bare = np.clip((0.4 - ndvi_) / 0.2, 0.0, 1.0)
    soil = np.clip(bsi_ / 0.1, 0.0, 1.0)
    return (bare * soil).astype(np.float32)


def new_disturbance_mask(
    p_t0: np.ndarray, p_t1: np.ndarray, p_high: float = 0.7, p_low: float = 0.3
) -> np.ndarray:
    """True where the t1 probability is high and the t0 probability is low.

    A pixel that was already mining at t0 is not new, and a pixel that never looked
    like mining at t1 is not an alert.
    """
    return (p_t1 >= p_high) & (p_t0 <= p_low)


def morphological_cleanup(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Remove single-pixel speckle (opening) and fill small pinholes (closing)."""
    structure = ndimage.generate_binary_structure(2, 1)
    out = ndimage.binary_opening(mask, structure=structure, iterations=iterations)
    out = ndimage.binary_closing(out, structure=structure, iterations=iterations)
    return out.astype(bool)


def min_area_filter(mask: np.ndarray, pixel_area_m2: float, min_area_ha: float) -> np.ndarray:
    """Remove connected patches smaller than min_area_ha hectares."""
    if min_area_ha <= 0:
        return mask.copy()
    labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    if n == 0:
        return mask.copy()
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    min_pixels = (min_area_ha * 10_000.0) / pixel_area_m2
    keep_ids = np.flatnonzero(sizes >= min_pixels) + 1
    return np.isin(labels, keep_ids)
