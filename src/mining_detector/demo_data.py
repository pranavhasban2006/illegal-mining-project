"""Synthetic Sentinel-2-like scenes for offline pipeline development.

Generates two time-step rasters (t0, t1) over a fake AOI with a patch of
"new mining disturbance" appearing at t1, plus a lease-boundary polygon that
covers only part of the disturbed area. This lets the whole O1-O5 pipeline
run end-to-end before real CDSE credentials/data are wired up.

Bands follow Sentinel-2 L2A ordering: B02 (blue), B03 (green), B04 (red),
B08 (NIR), B11 (SWIR1), B12 (SWIR2), each scaled 0-1 reflectance.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import box

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Fake AOI: a 300x300 pixel scene at 10m resolution, arbitrary UTM-like origin.
WIDTH, HEIGHT = 300, 300
PIXEL_SIZE = 10.0
ORIGIN_X, ORIGIN_Y = 500000.0, 9000000.0
CRS = "EPSG:32633"


def _base_scene(rng: np.random.Generator) -> np.ndarray:
    """Vegetated + riverine background scene, shape (bands, H, W)."""
    bands = np.zeros((6, HEIGHT, WIDTH), dtype=np.float32)

    # Healthy vegetation baseline reflectance with noise.
    bands[0] = rng.normal(0.05, 0.01, (HEIGHT, WIDTH))  # blue
    bands[1] = rng.normal(0.08, 0.01, (HEIGHT, WIDTH))  # green
    bands[2] = rng.normal(0.06, 0.01, (HEIGHT, WIDTH))  # red
    bands[3] = rng.normal(0.45, 0.03, (HEIGHT, WIDTH))  # NIR (high for veg)
    bands[4] = rng.normal(0.20, 0.02, (HEIGHT, WIDTH))  # SWIR1
    bands[5] = rng.normal(0.12, 0.02, (HEIGHT, WIDTH))  # SWIR2

    # A river running diagonally: low NIR, low SWIR (water absorbs both).
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    river = np.abs((xx * 0.6 + 40) - yy) < 6
    bands[3][river] = rng.normal(0.03, 0.005, river.sum())
    bands[4][river] = rng.normal(0.02, 0.005, river.sum())
    bands[5][river] = rng.normal(0.02, 0.005, river.sum())

    return np.clip(bands, 0, 1)


def _apply_bare_soil_patch(bands: np.ndarray, y0, y1, x0, x1, rng) -> np.ndarray:
    """Overwrite a rectangular patch with bare-soil/excavation spectral signature."""
    bands = bands.copy()
    h, w = y1 - y0, x1 - x0
    bands[0, y0:y1, x0:x1] = rng.normal(0.14, 0.02, (h, w))  # blue up
    bands[1, y0:y1, x0:x1] = rng.normal(0.18, 0.02, (h, w))  # green up
    bands[2, y0:y1, x0:x1] = rng.normal(0.22, 0.02, (h, w))  # red up (soil)
    bands[3, y0:y1, x0:x1] = rng.normal(0.20, 0.02, (h, w))  # NIR drops (no canopy)
    bands[4, y0:y1, x0:x1] = rng.normal(0.30, 0.02, (h, w))  # SWIR1 up (dry soil)
    bands[5, y0:y1, x0:x1] = rng.normal(0.26, 0.02, (h, w))  # SWIR2 up
    return np.clip(bands, 0, 1)


def generate_bitemporal_scene(seed: int = 42):
    """Returns dict with t0/t1 band stacks, transform, crs, and ground-truth masks."""
    rng = np.random.default_rng(seed)
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE)

    t0 = _base_scene(rng)
    # Pre-existing LICENSED mine already active at t0 (won't count as "new").
    t0 = _apply_bare_soil_patch(t0, 30, 70, 30, 70, rng)

    t1 = t0.copy()
    # Same licensed mine, slightly larger (normal permitted expansion).
    t1 = _apply_bare_soil_patch(t1, 25, 75, 25, 75, rng)
    # NEW illegal excavation appearing far from the licensed lease.
    t1 = _apply_bare_soil_patch(t1, 180, 230, 200, 250, rng)
    # A second, smaller new illegal site.
    t1 = _apply_bare_soil_patch(t1, 100, 120, 40, 65, rng)

    ground_truth_new_illegal = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    ground_truth_new_illegal[180:230, 200:250] = 1
    ground_truth_new_illegal[100:120, 40:65] = 1

    return {
        "t0": t0,
        "t1": t1,
        "transform": transform,
        "crs": CRS,
        "band_names": BAND_NAMES,
        "ground_truth_new_illegal": ground_truth_new_illegal,
    }


def make_lease_boundaries(out_path: str):
    """Licensed-lease polygon covering the pre-existing mine near (25-75, 25-75)."""
    x0 = ORIGIN_X + 20 * PIXEL_SIZE
    x1 = ORIGIN_X + 80 * PIXEL_SIZE
    y0 = ORIGIN_Y - 80 * PIXEL_SIZE
    y1 = ORIGIN_Y - 20 * PIXEL_SIZE
    lease = box(x0, y0, x1, y1)
    gdf = gpd.GeoDataFrame(
        {"lease_id": ["LIC-001"], "holder": ["Demo Mining Co."]},
        geometry=[lease],
        crs=CRS,
    )
    gdf.to_file(out_path, driver="GeoJSON")
    return out_path


def write_scene_geotiff(bands: np.ndarray, transform, crs, out_path: str):
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=bands.shape[1],
        width=bands.shape[2],
        count=bands.shape[0],
        dtype=bands.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(bands)
