"""Spectral indices computed from Sentinel-2 L2A band stacks.

Band stack convention throughout this project: shape (6, H, W), order
[B02, B03, B04, B08, B11, B12] = [blue, green, red, NIR, SWIR1, SWIR2],
reflectance scaled to 0-1.
"""

from __future__ import annotations

import numpy as np

BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = range(6)

_EPS = 1e-6


def ndvi(bands: np.ndarray) -> np.ndarray:
    """Vegetation index. Low/negative over bare soil, water, rock."""
    nir, red = bands[NIR], bands[RED]
    return (nir - red) / (nir + red + _EPS)


def ndmi(bands: np.ndarray) -> np.ndarray:
    """Moisture index. Low over dry disturbed soil; can spike over pit water."""
    nir, swir1 = bands[NIR], bands[SWIR1]
    return (nir - swir1) / (nir + swir1 + _EPS)


def bsi(bands: np.ndarray) -> np.ndarray:
    """Bare Soil Index. High over exposed soil / excavation surfaces."""
    red, swir1, nir, blue = bands[RED], bands[SWIR1], bands[NIR], bands[BLUE]
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue) + _EPS
    return num / den


def ndwi(bands: np.ndarray) -> np.ndarray:
    """Water index. Flags tailings ponds / flooded pits."""
    green, nir = bands[GREEN], bands[NIR]
    return (green - nir) / (green + nir + _EPS)


def mining_disturbance_index(bands: np.ndarray) -> np.ndarray:
    """Composite heuristic score in [-1, 1]: high => bare, dry, low-vegetation.

    Not a substitute for the learned classifier in detection.py -- used there
    as one input feature, and useful on its own for quick visual sanity checks.
    """
    veg = ndvi(bands)
    moist = ndmi(bands)
    bare = bsi(bands)
    return np.clip(bare - 0.5 * veg - 0.3 * moist, -1, 1)


def compute_feature_stack(bands: np.ndarray, terrain: dict | None = None) -> np.ndarray:
    """Stacks raw bands + derived indices into the classifier's feature space.

    terrain: optional {"elevation": (H,W) array, "dist_water": (H,W) array}
    -- real terrain covariates (see terrain.py), addressing a future-work
    item from the SAR-based mining paper reviewed in Project Review 1
    (persistent misclassification of peripheral/low-density mining
    structures; proposed fix was auxiliary elevation + hydrological-
    proximity covariates, since illegal mining is strongly correlated with
    stream proximity). Optional and backward compatible: omitting it
    reproduces the original 10-feature stack exactly.

    Returns shape (n_features, H, W).
    """
    feats = [
        bands[BLUE], bands[GREEN], bands[RED],
        bands[NIR], bands[SWIR1], bands[SWIR2],
        ndvi(bands), ndmi(bands), bsi(bands), ndwi(bands),
    ]
    if terrain:
        feats.append(terrain["elevation"])
        feats.append(terrain["dist_water"])
    return np.stack(feats, axis=0)


FEATURE_NAMES = [
    "B02", "B03", "B04", "B08", "B11", "B12",
    "NDVI", "NDMI", "BSI", "NDWI",
]

TERRAIN_FEATURE_NAMES = ["ELEVATION", "DIST_WATER"]
