"""O4: Cross-reference detected change against licensed-mining-lease boundaries.

Produces a per-pixel "suspicious" mask (new disturbance NOT inside any
known lease polygon) plus a confidence score blending classifier
probability with distance-outside-lease.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from affine import Affine


def rasterize_leases(
    lease_gdf: gpd.GeoDataFrame,
    out_shape: tuple[int, int],
    transform: Affine,
    target_crs: str,
) -> np.ndarray:
    """Burns lease polygons onto the scene grid. Returns (H,W) bool: True=licensed."""
    if lease_gdf is None or len(lease_gdf) == 0:
        return np.zeros(out_shape, dtype=bool)

    gdf = lease_gdf.to_crs(target_crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros(out_shape, dtype=bool)

    burned = rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8
    )
    return burned.astype(bool)


def suspicious_area_mask(
    new_disturbance_mask: np.ndarray,
    licensed_mask: np.ndarray,
) -> np.ndarray:
    """New disturbance that falls outside every known licensed lease."""
    return (new_disturbance_mask.astype(bool) & ~licensed_mask).astype(np.uint8)


def suspicious_confidence(
    suspicious_mask: np.ndarray,
    detection_proba: np.ndarray,
    change_conf: np.ndarray,
) -> np.ndarray:
    """Confidence score in [0,1] per pixel: only meaningful where suspicious_mask==1.

    Blends classifier confidence and magnitude of temporal change -- a pixel
    that is both strongly classified as mining AND shows a large jump in
    probability since t0 scores highest.
    """
    score = 0.6 * detection_proba + 0.4 * change_conf
    return np.where(suspicious_mask.astype(bool), score, 0.0)
