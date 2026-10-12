"""Spectral indices and the per-pixel feature stack.

Each index is a simple ratio of two Sentinel-2 bands. Together they separate
vegetation (high NDVI), bare or excavated soil (high BSI, low NDVI) and water
(high NDWI).

Bands are expected as reflectance in the range 0 to 1 (Sentinel-2 values already
divided by 10,000), keyed by band name, for example {"B04": array, "B08": array}.
"""

from __future__ import annotations

import numpy as np

# Small number that keeps the ratios finite when both bands are near zero.
_EPS = 1e-6

FEATURE_NAMES = ("B02", "B03", "B04", "B08", "B11", "B12", "NDVI", "NDMI", "BSI", "NDWI")


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return (a - b) / (a + b), with a safe denominator."""
    return (a - b) / (a + b + _EPS)


def ndvi(b08: np.ndarray, b04: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. Near 1 for dense forest, near 0 for bare soil."""
    return _normalized_difference(b08, b04)


def ndmi(b08: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index. High values mean moist canopy or soil."""
    return _normalized_difference(b08, b11)


def ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index. High values flag open water such as rivers and ponds."""
    return _normalized_difference(b03, b08)


def bsi(b02: np.ndarray, b04: np.ndarray, b08: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Bare Soil Index. High values mean exposed soil, which is what pits and tailings look like."""
    num = (b11 + b04) - (b08 + b02)
    den = (b11 + b04) + (b08 + b02)
    return num / (den + _EPS)


def compute_feature_stack(bands: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Stack the six bands and four indices into one (H, W, F) float32 array.

    Args:
        bands: mapping from band name ("B02", "B03", "B04", "B08", "B11", "B12") to
            2-D reflectance arrays of identical shape.

    Returns:
        (stack, names): stack has shape (H, W, 10); names lists the feature in the
        same order as the last axis.

    Raises:
        KeyError: if a required band is missing.
        ValueError: if the band arrays do not share one shape.
    """
    required = ("B02", "B03", "B04", "B08", "B11", "B12")
    missing = [b for b in required if b not in bands]
    if missing:
        raise KeyError(f"Missing Sentinel-2 bands: {', '.join(missing)}")

    shapes = {bands[b].shape for b in required}
    if len(shapes) != 1:
        raise ValueError(f"Band arrays have different shapes: {sorted(shapes)}")

    b = {k: np.asarray(bands[k], dtype=np.float32) for k in required}
    layers = [
        b["B02"], b["B03"], b["B04"], b["B08"], b["B11"], b["B12"],
        ndvi(b["B08"], b["B04"]),
        ndmi(b["B08"], b["B11"]),
        bsi(b["B02"], b["B04"], b["B08"], b["B11"]),
        ndwi(b["B03"], b["B08"]),
    ]
    stack = np.stack(layers, axis=-1).astype(np.float32)
    return stack, list(FEATURE_NAMES)
