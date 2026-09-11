"""O3: Bi-temporal change detection.

Takes classified mining-probability maps from two observation dates and
isolates pixels that newly became disturbed (or expanded), filtering out
noise via connected-component size thresholding.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def new_disturbance_mask(
    proba_t0: np.ndarray,
    proba_t1: np.ndarray,
    threshold: float = 0.5,
    min_blob_pixels: int = 15,
) -> np.ndarray:
    """Returns binary mask (H,W) of pixels newly classified as mining at t1.

    A pixel counts as "new" if it crosses the detection threshold at t1 but
    was below it at t0 -- this suppresses already-known (t0) mining areas
    and keeps only genuine spatio-temporal change.
    """
    mask_t0 = proba_t0 >= threshold
    mask_t1 = proba_t1 >= threshold
    new_mask = mask_t1 & ~mask_t0

    labeled, n_blobs = ndimage.label(new_mask)
    if n_blobs == 0:
        return new_mask.astype(np.uint8)

    sizes = ndimage.sum(new_mask, labeled, index=range(1, n_blobs + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_blob_pixels]
    cleaned = np.isin(labeled, keep_labels)
    return cleaned.astype(np.uint8)


def change_confidence(proba_t0: np.ndarray, proba_t1: np.ndarray) -> np.ndarray:
    """Magnitude of increase in mining probability between the two dates."""
    return np.clip(proba_t1 - proba_t0, 0, 1)
