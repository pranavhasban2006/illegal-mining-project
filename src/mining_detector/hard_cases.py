"""Selects a stratified sample of genuinely ambiguous pixel locations for
human labeling, to validate the classifier beyond the "easy" extremes it
was trained/tested on (see README's validation-gap discussion).

Categories deliberately target known confusable cases:
  - boundary_zone: just outside a concession, still looks disturbed --
    could be real illegal expansion, or just naturally sparse vegetation.
  - river_sandbar: bright, bare-looking pixels near water -- classic
    false-positive risk (natural sediment mimics excavated soil).
  - classifier_uncertain: model's predicted probability near 0.5 -- the
    literal cases the model itself is least sure about.
  - far_bare_random: looks disturbed AND far from any known concession --
    candidate *real* undiscovered mining, or just a natural bare patch.
  - ambiguous_vegetation: mid-range NDVI far from concessions -- fallow
    land / secondary growth / agriculture, easily confused with either class.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import indices


def _random_qualifying_points(mask: np.ndarray, n: int, min_separation: int, rng):
    """Picks up to n points from mask==True, spaced >= min_separation pixels apart."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    order = rng.permutation(len(ys))
    chosen = []
    for idx in order:
        y, x = ys[idx], xs[idx]
        if all((y - cy) ** 2 + (x - cx) ** 2 >= min_separation ** 2 for cy, cx in chosen):
            chosen.append((y, x))
        if len(chosen) >= n:
            break
    return chosen


def select_hard_cases(
    bands: np.ndarray,
    concession_mask: np.ndarray,
    proba: np.ndarray,
    n_per_category: int = 8,
    seed: int = 0,
) -> list[dict]:
    """Returns list of {category, row, col} for candidate chip extraction."""
    rng = np.random.default_rng(seed)
    ndvi = indices.ndvi(bands)
    ndwi = indices.ndwi(bands)
    bsi = indices.bsi(bands)

    # Exclude Sentinel-2 tile-boundary nodata gaps (all-zero pixels) from
    # every candidate pool -- these are missing data, not real land cover.
    valid = bands.sum(axis=0) > 0

    pixel_dist_to_concession = ndimage.distance_transform_edt(~concession_mask)

    cases = []

    # boundary_zone: 0-30px (~300m at 10m/px) outside a concession, disturbed-looking.
    boundary = valid & (pixel_dist_to_concession > 0) & (pixel_dist_to_concession <= 30) & (bsi > np.nanpercentile(bsi, 60))
    for y, x in _random_qualifying_points(boundary, n_per_category, 20, rng):
        cases.append({"category": "boundary_zone", "row": int(y), "col": int(x)})

    # river_sandbar: bright in all visible bands, near water (nearby NDWI variance high).
    bright = (bands[0] > np.nanpercentile(bands[0], 85)) & (bands[1] > np.nanpercentile(bands[1], 85))
    near_water = ndimage.maximum_filter(ndwi, size=9) > 0.1
    sandbar = valid & bright & near_water
    for y, x in _random_qualifying_points(sandbar, n_per_category, 20, rng):
        cases.append({"category": "river_sandbar", "row": int(y), "col": int(x)})

    # classifier_uncertain: proba near 0.5, far from any concession (not already labeled).
    uncertain = valid & (np.abs(proba - 0.5) < 0.05) & (pixel_dist_to_concession > 30)
    for y, x in _random_qualifying_points(uncertain, n_per_category, 20, rng):
        cases.append({"category": "classifier_uncertain", "row": int(y), "col": int(x)})

    # far_bare_random: looks disturbed, far (>50px ~500m) from any concession.
    far_bare = valid & (bsi > np.nanpercentile(bsi, 90)) & (pixel_dist_to_concession > 50)
    for y, x in _random_qualifying_points(far_bare, n_per_category, 20, rng):
        cases.append({"category": "far_bare_random", "row": int(y), "col": int(x)})

    # ambiguous_vegetation: mid-range NDVI, far from concessions.
    lo, hi = np.nanpercentile(ndvi, [40, 60])
    ambiguous_veg = valid & (ndvi >= lo) & (ndvi <= hi) & (pixel_dist_to_concession > 30)
    for y, x in _random_qualifying_points(ambiguous_veg, n_per_category, 20, rng):
        cases.append({"category": "ambiguous_vegetation", "row": int(y), "col": int(x)})

    return cases


def extract_rgb_chip(bands: np.ndarray, row: int, col: int, chip_size: int = 64) -> np.ndarray:
    """Returns an (chip_size, chip_size, 3) uint8 true-color RGB chip centered at (row, col).

    Uses a simple percentile stretch per-chip so each thumbnail is
    individually well-exposed regardless of scene-wide brightness.
    """
    half = chip_size // 2
    h, w = bands.shape[1:]
    r0, r1 = max(row - half, 0), min(row + half, h)
    c0, c1 = max(col - half, 0), min(col + half, w)

    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)  # R=B04, G=B03, B=B02
    chip = rgb[r0:r1, c0:c1, :]

    out = np.zeros((chip_size, chip_size, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(chip, [2, 98])
    stretched = np.clip((chip - lo) / (hi - lo + 1e-6), 0, 1)
    out[: chip.shape[0], : chip.shape[1], :] = (stretched * 255).astype(np.uint8)
    return out
