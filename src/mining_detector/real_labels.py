"""Real ground-truth labels for O2, sourced from RAISG's mining-concessions
dataset instead of the percentile-based weak labels in weak_labels.py.

Key idea: legal and illegal mining are spectrally indistinguishable (both
are excavated bare soil/tailings) -- the difference is administrative, not
spectral. So real, government-sourced concession polygons make a genuine
positive-label source for "this pixel has a mining spectral signature",
while pixels far from any concession that still look forested make a
genuine negative source. This directly replaces the self-referential
weak-label bootstrap (see README's "weak-label ceiling" limitation) with
labels grounded in an external, independently-collected dataset.

O4 then reuses the SAME concession polygons to flag detected mining-
signature pixels outside any active concession as suspicious.
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd
from scipy import ndimage

from . import indices, weak_labels
from .licensing import rasterize_leases

UNLABELED = weak_labels.UNLABELED
NOT_MINING = weak_labels.NOT_MINING
MINING = weak_labels.MINING

# Only these statuses represent a currently active/granted legal right to
# mine. "Bloqueado" (frozen/disputed), "Extinguido" (expired/revoked) and
# "Tramite" (pending application) do not.
ACTIVE_STATUSES = {"Titulada", "Explotación", "Explotacion"}


def active_concessions(concessions_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "situacion" not in concessions_gdf.columns:
        return concessions_gdf
    return concessions_gdf[concessions_gdf["situacion"].isin(ACTIVE_STATUSES)]


def generate_labels_from_concessions(
    bands: np.ndarray,
    transform,
    crs,
    concessions_gdf: gpd.GeoDataFrame,
    within_concession_percentile: float = 40.0,
    negative_buffer_m: float = 300.0,
    negative_ndvi_percentile: float = 75.0,
) -> np.ndarray:
    """Returns (H,W) int8 labels: 1=mining, 0=not-mining, -1=unlabeled.

    Positive: pixels inside an active concession AND above-median
    disturbance score within that concession (guards against concession
    polygons that are only partially exploited -- not every square meter of
    a titled concession is necessarily dug up yet).
    Negative: pixels more than `negative_buffer_m` from ANY concession
    (active or not, as a conservative exclusion zone) AND clearly dense
    vegetation (top NDVI percentile).
    """
    active = active_concessions(concessions_gdf)
    concession_mask = rasterize_leases(active, bands.shape[1:], transform, crs)

    disturbance = weak_labels._raw_disturbance_score(bands)
    ndvi = indices.ndvi(bands)

    labels = np.full(bands.shape[1:], UNLABELED, dtype=np.int8)

    if concession_mask.any():
        within_thresh = np.nanpercentile(disturbance[concession_mask], within_concession_percentile)
        labels[concession_mask & (disturbance >= within_thresh)] = MINING

    # Distance (in meters) from every pixel to the nearest concession of ANY
    # status -- used as a conservative exclusion buffer for negative labels,
    # since even a frozen/pending concession signals "known mining interest
    # here", not confidently untouched forest.
    all_concession_mask = rasterize_leases(concessions_gdf, bands.shape[1:], transform, crs)
    pixel_size_m = abs(transform.a)
    dist_px = ndimage.distance_transform_edt(~all_concession_mask)
    dist_m = dist_px * pixel_size_m

    ndvi_thresh = np.nanpercentile(ndvi, negative_ndvi_percentile)
    negative_mask = (dist_m >= negative_buffer_m) & (ndvi >= ndvi_thresh)
    labels[negative_mask & (labels == UNLABELED)] = NOT_MINING

    return labels
