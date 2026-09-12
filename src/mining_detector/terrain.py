"""O2 feature engineering: terrain covariates (elevation, distance to
hydrology), addressing a future-work item named explicitly in the SAR-based
mining-monitoring paper reviewed in Project Review 1 -- the authors found
persistent misclassification around peripheral/low-density mining
structures and proposed "integrating auxiliary terrain covariates
(elevation, proximity to hydrological features)" to constrain the search
space, since illegal mining is strongly correlated with stream proximity.

Elevation comes from the Copernicus DEM GLO-30 (real ~30m global DEM,
public AWS Open Data, no auth/API key needed). Distance-to-water is
derived from the Sentinel-2 scene's own NDWI, not the DEM (a global DEM
has no reliable water mask at this resolution) -- reusing the same
distance-transform pattern already used for concession-distance in
real_labels.py.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.env import Env
from scipy import ndimage

from . import indices

DEM_BUCKET = "copernicus-dem-30m"


def _dem_tile_keys(bbox_wgs84: tuple[float, float, float, float]) -> list[str]:
    """1x1 degree Copernicus DEM tile keys covering a WGS84 bbox."""
    minx, miny, maxx, maxy = bbox_wgs84
    lat0, lat1 = math.floor(miny), math.floor(maxy)
    lon0, lon1 = math.floor(minx), math.floor(maxx)

    keys = []
    for lat in range(lat0, lat1 + 1):
        ns = f"S{abs(lat):02d}" if lat < 0 else f"N{lat:02d}"
        for lon in range(lon0, lon1 + 1):
            ew = f"W{abs(lon):03d}" if lon < 0 else f"E{lon:03d}"
            name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
            keys.append(f"/vsis3/{DEM_BUCKET}/{name}/{name}.tif")
    return keys


def fetch_elevation(
    bbox_wgs84: tuple[float, float, float, float],
    target_shape: tuple[int, int],
    target_transform,
    target_crs: str,
) -> np.ndarray:
    """Returns (H,W) elevation in meters, resampled onto the given grid.

    Reads only the windowed extent needed from each Copernicus DEM COG tile
    (no full-tile download) via GDAL's /vsis3/ virtual filesystem with
    unsigned S3 access (public dataset).
    """
    keys = _dem_tile_keys(bbox_wgs84)

    with Env(AWS_NO_SIGN_REQUEST="YES", AWS_REGION="eu-central-1"):
        # merge() opens each source, reads only the region overlapping
        # `bounds`, and mosaics -- a full 43MB tile is never pulled.
        srcs = [rasterio.open(k) for k in keys]
        try:
            mosaic, mosaic_transform = merge(srcs, bounds=bbox_wgs84)
        finally:
            for s in srcs:
                s.close()

    mosaic = mosaic[0].astype(np.float32)  # (H, W), single band

    dest = np.zeros(target_shape, dtype=np.float32)
    reproject(
        source=mosaic,
        destination=dest,
        src_transform=mosaic_transform,
        src_crs="EPSG:4326",
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear,
    )
    return dest


def compute_distance_to_water(bands: np.ndarray, pixel_size_m: float, ndwi_threshold: float = 0.0) -> np.ndarray:
    """Returns (H,W) distance-to-nearest-water in meters, from the scene's
    own NDWI (thresholded) rather than the DEM -- a 30m global DEM has no
    reliable water layer at this scale, but Sentinel-2 does."""
    water_mask = indices.ndwi(bands) > ndwi_threshold
    if not water_mask.any():
        # No water detected in-scene -- cap at the scene's own diagonal
        # extent rather than returning inf, which would break any
        # downstream scaler/classifier fed this as a feature.
        h, w = bands.shape[1:]
        cap = float(np.hypot(h, w) * pixel_size_m)
        return np.full((h, w), cap, dtype=np.float32)
    dist_px = ndimage.distance_transform_edt(~water_mask)
    return (dist_px * pixel_size_m).astype(np.float32)
