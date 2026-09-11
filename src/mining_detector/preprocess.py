"""O1: Preprocess downloaded per-band GeoTIFFs into the canonical band stack.

Reads individual band rasters (as produced by download.py), resamples them
all to a common 10m grid, and stacks them into the (6,H,W) array format
consumed by indices.py / detection.py.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .download import REQUIRED_BANDS


def load_band_stack(band_paths: dict, target_resolution: float = 10.0):
    """band_paths: {band_name: path_to_tif}. Returns (bands, transform, crs)."""
    arrays = []
    ref_transform = None
    ref_crs = None
    ref_shape = None

    for band in REQUIRED_BANDS:
        with rasterio.open(band_paths[band]) as src:
            if ref_transform is None:
                # First band (assumed 10m, e.g. B02) sets the target grid.
                ref_transform = src.transform
                ref_crs = src.crs
                ref_shape = (src.height, src.width)
                data = src.read(1)
            else:
                with WarpedVRT(
                    src,
                    crs=ref_crs,
                    transform=ref_transform,
                    width=ref_shape[1],
                    height=ref_shape[0],
                    resampling=Resampling.bilinear,
                ) as vrt:
                    data = vrt.read(1)

            arrays.append(data.astype(np.float32) / 10000.0)  # L2A scale factor

    bands = np.stack(arrays, axis=0)
    return bands, ref_transform, ref_crs


def crop_to_bbox(bands: np.ndarray, transform, crs, bbox_wgs84: tuple[float, float, float, float]):
    """Crops an in-memory band stack to a WGS84 bbox. Returns (bands, transform)."""
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds, transform as window_transform

    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", crs, *bbox_wgs84)
    window = from_bounds(minx, miny, maxx, maxy, transform=transform).round_lengths().round_offsets()

    row_off = max(int(window.row_off), 0)
    col_off = max(int(window.col_off), 0)
    row_end = min(row_off + int(window.height), bands.shape[1])
    col_end = min(col_off + int(window.width), bands.shape[2])

    cropped = bands[:, row_off:row_end, col_off:col_end]
    from rasterio.windows import Window
    actual_window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
    new_transform = window_transform(actual_window, transform)
    return cropped, new_transform


def clip_to_aoi(bands: np.ndarray, transform, aoi_geom, crs):
    """Masks the band stack to an AOI polygon (shapely geometry in `crs`)."""
    from rasterio.features import geometry_mask

    mask = geometry_mask(
        [aoi_geom], out_shape=bands.shape[1:], transform=transform, invert=True
    )
    clipped = bands.copy()
    clipped[:, ~mask] = np.nan
    return clipped
