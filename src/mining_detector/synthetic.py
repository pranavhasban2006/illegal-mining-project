"""Synthetic two-date Sentinel-2-like scene with a known answer.

The scene contains:
    - dense forest and a river (both dates),
    - one existing, licensed mine (bare at both dates, covered by an active concession),
    - two NEW unlicensed mining sites that appear only at t1.

Because the answer is known, the pipeline can be checked end-to-end without internet.
The demo and the tests use this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

# UTM zone 19S covers the default Peru AOI.
SYNTH_CRS = "EPSG:32719"
PIXEL_M = 10.0

# Typical reflectance per land cover (Sentinel-2 scale, 0-1).
_SPECTRA = {
    "forest": {"B02": 0.03, "B03": 0.06, "B04": 0.04, "B08": 0.35, "B11": 0.15, "B12": 0.08},
    "bare":   {"B02": 0.12, "B03": 0.14, "B04": 0.25, "B08": 0.28, "B11": 0.35, "B12": 0.30},
    "water":  {"B02": 0.05, "B03": 0.08, "B04": 0.04, "B08": 0.02, "B11": 0.01, "B12": 0.01},
}


@dataclass
class SyntheticScene:
    """Container for the synthetic data and its ground truth."""

    bands_t0: dict[str, np.ndarray]
    bands_t1: dict[str, np.ndarray]
    transform: Affine
    crs: str
    concessions: gpd.GeoDataFrame
    truth_new_mask: np.ndarray           # True where NEW mining appeared at t1
    licensed_mask: np.ndarray            # True where the licensed (pre-existing) mine is
    new_site_boxes: list = field(default_factory=list)  # shapely boxes, map coords
    t0: str = "2024-07-27"
    t1: str = "2025-07-27"

    @property
    def shape(self) -> tuple[int, int]:
        return self.truth_new_mask.shape


def _fill(canvas: dict[str, np.ndarray], mask: np.ndarray, spectrum: dict[str, float]) -> None:
    """Paint a land cover into the bands wherever mask is True."""
    for band, value in spectrum.items():
        canvas[band][mask] = value


def make_synthetic_scene(seed: int = 42, size: int = 220) -> SyntheticScene:
    """Build the synthetic scene.

    Args:
        seed: random seed for the noise, so the result is reproducible.
        size: side length of the square raster in pixels (10 m each).

    Returns:
        A SyntheticScene with two dates, concessions and ground truth.
    """
    rng = np.random.default_rng(seed)
    h = w = size
    origin_x, origin_y = 500_000.0, 8_600_000.0  # top-left corner in UTM metres
    transform = Affine(PIXEL_M, 0, origin_x, 0, -PIXEL_M, origin_y)

    rows, cols = np.mgrid[0:h, 0:w]

    forest = np.ones((h, w), dtype=bool)
    river = (rows >= 150) & (rows < 165)              # horizontal river near the bottom
    forest &= ~river

    licensed = (rows >= 40) & (rows < 80) & (cols >= 30) & (cols < 70)   # 40x40 px = 16 ha
    site_a = (rows >= 100) & (rows < 130) & (cols >= 170) & (cols < 200) # 30x30 px = 9 ha, >500 m from licence
    site_b = (rows >= 50) & (rows < 70) & (cols >= 150) & (cols < 175)   # 20x25 px = 5 ha

    def base_scene(with_new: bool) -> dict[str, np.ndarray]:
        bands = {b: np.zeros((h, w), dtype=np.float32) for b in _SPECTRA["forest"]}
        _fill(bands, forest, _SPECTRA["forest"])
        _fill(bands, river, _SPECTRA["water"])
        _fill(bands, licensed, _SPECTRA["bare"])
        if with_new:
            _fill(bands, site_a | site_b, _SPECTRA["bare"])
        for b in bands:
            bands[b] = np.clip(bands[b] + rng.normal(0, 0.005, (h, w)).astype(np.float32), 0.0, 1.0)
        return bands

    bands_t0 = base_scene(with_new=False)
    bands_t1 = base_scene(with_new=True)

    new_mask = site_a | site_b
    new_boxes = []
    for region in (site_a, site_b):
        r = np.where(region.any(axis=1))[0]
        c = np.where(region.any(axis=0))[0]
        x0 = origin_x + c[0] * PIXEL_M
        x1 = origin_x + (c[-1] + 1) * PIXEL_M
        y1 = origin_y - r[0] * PIXEL_M
        y0 = origin_y - (r[-1] + 1) * PIXEL_M
        new_boxes.append(box(x0, y0, x1, y1))

    # Licence covers the licensed mine only, with a margin, so the two new sites are illegal.
    lx0 = origin_x + 30 * PIXEL_M - 50
    lx1 = origin_x + 70 * PIXEL_M + 50
    ly1 = origin_y - 40 * PIXEL_M + 50
    ly0 = origin_y - 80 * PIXEL_M - 50
    concessions = gpd.GeoDataFrame(
        {"name": ["Licensed mine"], "status": ["active"]},
        geometry=[box(lx0, ly0, lx1, ly1)],
        crs=SYNTH_CRS,
    )

    return SyntheticScene(
        bands_t0=bands_t0,
        bands_t1=bands_t1,
        transform=transform,
        crs=SYNTH_CRS,
        concessions=concessions,
        truth_new_mask=new_mask,
        licensed_mask=licensed,
        new_site_boxes=new_boxes,
    )
