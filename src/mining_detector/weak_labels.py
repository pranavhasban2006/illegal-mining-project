"""Bootstrap weak labels from spectral thresholds, for training detection.py
when no hand-annotated ground truth exists yet.

This is a documented weak-supervision step, not a substitute for real
labels: it trades some accuracy for zero manual annotation effort, and
should be swapped for hand-digitized or open-dataset labels (see README)
once available. Thresholds are percentile-based on the scene itself rather
than fixed absolute values, since Sentinel-2 reflectance distributions vary
scene-to-scene with atmosphere/season/terrain.
"""

from __future__ import annotations

import numpy as np

from . import indices

UNLABELED = -1
NOT_MINING = 0
MINING = 1


def _raw_disturbance_score(bands: np.ndarray) -> np.ndarray:
    """Unclipped version of indices.mining_disturbance_index.

    Percentile-based thresholding needs the raw, unclipped score: the
    clipped version saturates at +-1 for extreme pixels, which on a heavily
    disturbed scene (e.g. La Pampa) can tie 30%+ of pixels at the same
    capped value and silently break percentile selection (a threshold meant
    to select the top 2% ends up selecting every tied pixel instead).
    """
    veg = indices.ndvi(bands)
    moist = indices.ndmi(bands)
    bare = indices.bsi(bands)
    return bare - 0.5 * veg - 0.3 * moist


def generate_weak_labels(
    bands: np.ndarray,
    positive_percentile: float = 98.0,
    negative_percentile: float = 95.0,
) -> np.ndarray:
    """Returns (H,W) int8 array: 1=confident mining, 0=confident not-mining,
    -1=unlabeled (excluded from training -- ambiguous pixels).

    Positive: top `positive_percentile` of the composite disturbance index
    (bare, dry, low-vegetation signature).
    Negative: top `negative_percentile` of NDVI (confidently dense vegetation)
    among pixels the disturbance index rates as NOT disturbed.
    """
    mdi = _raw_disturbance_score(bands)
    ndvi = indices.ndvi(bands)

    labels = np.full(mdi.shape, UNLABELED, dtype=np.int8)

    positive_thresh = np.nanpercentile(mdi, positive_percentile)
    labels[mdi >= positive_thresh] = MINING

    low_disturbance_thresh = np.nanpercentile(mdi, 100 - negative_percentile)
    ndvi_thresh = np.nanpercentile(ndvi, negative_percentile)
    negative_mask = (mdi <= low_disturbance_thresh) & (ndvi >= ndvi_thresh)
    labels[negative_mask & (labels == UNLABELED)] = NOT_MINING

    return labels


def label_summary(labels: np.ndarray) -> dict:
    total = labels.size
    n_mining = int((labels == MINING).sum())
    n_not = int((labels == NOT_MINING).sum())
    n_unlabeled = int((labels == UNLABELED).sum())
    return {
        "n_mining": n_mining,
        "n_not_mining": n_not,
        "n_unlabeled": n_unlabeled,
        "pct_labeled": 100.0 * (n_mining + n_not) / total,
    }
