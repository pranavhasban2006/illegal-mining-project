"""Training labels from public data, never from the classifier's own output.

Positive (mining) pixels come from the Global Mining Footprint polygons (Maus et al. 2022),
clipped to the area of interest. If that file is missing, concession polygons are used
as a weaker fallback and a caveat says so.

Negative (not mining) pixels are:
    - dense vegetation: NDVI > 0.6 and more than 300 m from any mining polygon, and
    - permanent water: NDWI above a water threshold.

Optional hand-labelled points (GeoJSON or CSV with lon, lat, label 0/1) override the
automatic labels in a small neighbourhood around each point. This is how review
decisions from the app feed back into training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from scipy import ndimage
from shapely.geometry import box

from .preprocess import Grid

log = logging.getLogger(__name__)

GMF_RELATIVE_PATH = Path("global_mining") / "mining_polygons.gpkg"
UNLABELED = -1
NEG_VEG_NDVI = 0.6
NEG_BUFFER_M = 300.0
WATER_NDWI = 0.3

FETCH_HINT = (
    "No mining training labels were found. Run this once to download them:\n"
    "    python scripts/fetch_datasets.py mining\n"
    "or pass --concessions with a concession file as a weaker fallback."
)


@dataclass
class LabelSet:
    """Per-pixel labels: 1 = mining, 0 = not mining, -1 = unlabelled."""

    y: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def n_pos(self) -> int:
        return int((self.y == 1).sum())

    @property
    def n_neg(self) -> int:
        return int((self.y == 0).sum())


MAX_FOOTPRINT_KM2 = 100.0  # the median mine footprint is about 0.1 km2; larger ones are suspect


def _polygons_in_grid(path: Path, grid: Grid, aoi_bbox_grid: tuple) -> tuple[gpd.GeoDataFrame, int]:
    """Load polygons, reproject them to the grid CRS, drop implausibly large ones, then clip.

    Returns:
        (clipped polygons, number of oversize polygons that were dropped).
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # Measure size in an equal-area projection; a UTM zone would distort polygons that
    # span several zones.
    area_km2 = gdf.to_crs("EPSG:6933").geometry.area / 1e6
    oversize = (area_km2 > MAX_FOOTPRINT_KM2).to_numpy()
    gdf = gdf.to_crs(grid.crs)
    n_dropped = int(oversize.sum())
    if n_dropped:
        log.warning("Dropping %d mining polygon(s) larger than %.0f km2 (likely data errors)",
                    n_dropped, MAX_FOOTPRINT_KM2)
        gdf = gdf[~oversize]
    clip = gpd.GeoDataFrame(geometry=[box(*aoi_bbox_grid)], crs=grid.crs)
    return gpd.clip(gdf, clip), n_dropped


def _rasterize(geoms, grid: Grid) -> np.ndarray:
    """Burn polygons onto the grid. A pixel counts if the polygon touches it."""
    if len(geoms) == 0:
        return np.zeros(grid.shape, dtype=bool)
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(grid.shape, dtype=bool)
    burned = features.rasterize(
        shapes, out_shape=grid.shape, transform=grid.transform,
        fill=0, dtype="uint8", all_touched=True,
    )
    return burned.astype(bool)


def _grid_bounds(grid: Grid) -> tuple[float, float, float, float]:
    h, w = grid.shape
    t = grid.transform
    return (t.c, t.f + h * t.e, t.c + w * t.a, t.f)


def find_positive_source(external_dir: Path, concessions_path: Path | None) -> tuple[Path, str]:
    """Pick the positive-label file: mining footprint first, concessions second.

    Raises:
        FileNotFoundError: with a plain-English instruction if neither exists.
    """
    gmf = Path(external_dir) / GMF_RELATIVE_PATH
    if gmf.exists():
        return gmf, "global_mining_footprint"
    if concessions_path is not None and Path(concessions_path).exists():
        return Path(concessions_path), "concessions_fallback"
    raise FileNotFoundError(FETCH_HINT)


def build_labels(
    grid: Grid,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    aoi_bbox_wgs84: tuple[float, float, float, float],
    external_dir: str | Path = "data/external",
    concessions_path: str | Path | None = None,
    extra_labels_path: str | Path | None = None,
    neighborhood_px: int = 0,
) -> LabelSet:
    """Create the per-pixel training labels for one area.

    Args:
        grid: the shared target grid.
        ndvi, ndwi: index arrays on that grid (used for negatives only).
        aoi_bbox_wgs84: area of interest (west, south, east, north), for clipping.
        external_dir: folder holding global_mining/mining_polygons.gpkg.
        concessions_path: optional fallback positives (concession polygons).
        extra_labels_path: optional GeoJSON/CSV with lon, lat, label (0/1) points.
        neighborhood_px: side length of the square around each hand label that it
            overrides (0 means only the pixel under the point).

    Returns:
        LabelSet with the label array and counts for each class.
    """
    from pyproj import Transformer

    if ndvi.shape != grid.shape or ndwi.shape != grid.shape:
        raise ValueError("ndvi and ndwi must match the grid shape.")

    source_path, source_name = find_positive_source(Path(external_dir), concessions_path)
    notes: list[str] = []
    if source_name == "concessions_fallback":
        notes.append(
            "Positive labels are concession polygons, not a mining footprint. Concessions "
            "include forest and unmined land, so this is a weak, noisy label source."
        )

    # Area in the grid CRS, for clipping.
    to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    w, s, e, n = aoi_bbox_wgs84
    xs, ys = to_grid.transform([w, e, w, e], [s, s, n, n])
    aoi_grid = (min(xs), min(ys), max(xs), max(ys))

    polys, n_oversize = _polygons_in_grid(source_path, grid, aoi_grid)
    if n_oversize:
        notes.append(
            f"{n_oversize} mining polygon(s) larger than {MAX_FOOTPRINT_KM2:.0f} km2 were ignored "
            "as probable data errors."
        )
    pos = _rasterize(list(polys.geometry), grid)
    if not pos.any():
        raise ValueError(
            f"No {source_name.replace('_', ' ')} polygons overlap this area. "
            "Choose an area with known mining, for example the default example area."
        )

    # Distance (metres) from every pixel to the nearest positive pixel.
    dist_m = ndimage.distance_transform_edt(~pos) * abs(grid.transform.a)

    y = np.full(grid.shape, UNLABELED, dtype=np.int8)
    water = (ndwi > WATER_NDWI) & ~pos
    dense_veg = (ndvi > NEG_VEG_NDVI) & (dist_m > NEG_BUFFER_M) & ~pos & ~water

    y[dense_veg] = 0
    y[water] = 0
    y[pos] = 1

    counts = {
        "positive_mining_px": int(pos.sum()),
        "negative_dense_vegetation_px": int(dense_veg.sum()),
        "negative_water_px": int(water.sum()),
        "hand_labels_px": 0,
        "oversize_polygons_dropped": n_oversize,
    }

    if extra_labels_path is not None:
        counts["hand_labels_px"] = _apply_hand_labels(y, grid, extra_labels_path, neighborhood_px)

    log.info("Labels: %s", counts)
    return LabelSet(y=y, counts=counts, source=source_name, notes=notes)


def _apply_hand_labels(y: np.ndarray, grid: Grid, path: str | Path, neighborhood_px: int) -> int:
    """Stamp hand labels (0 or 1) into y, overriding automatic labels. Returns pixels set."""
    from pyproj import Transformer

    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if not {"lon", "lat", "label"}.issubset(df.columns):
            raise ValueError("Hand-label CSV needs columns: lon, lat, label")
        pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    else:
        pts = gpd.read_file(path)
        if "label" not in pts.columns:
            raise ValueError("Hand-label file needs a 'label' column with 0 or 1.")
    pts = pts[pts["label"].isin([0, 1])]
    if pts.empty:
        return 0

    pts = pts.to_crs(grid.crs)
    stamped = 0
    r_half = neighborhood_px // 2
    for geom, lab in zip(pts.geometry, pts["label"].astype(int)):
        col = int(np.floor((geom.x - grid.transform.c) / grid.transform.a))
        row = int(np.floor((geom.y - grid.transform.f) / grid.transform.e))
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            continue
        r0, r1 = max(0, row - r_half), min(grid.shape[0], row + r_half + 1)
        c0, c1 = max(0, col - r_half), min(grid.shape[1], col + r_half + 1)
        y[r0:r1, c0:c1] = lab
        stamped += (r1 - r0) * (c1 - c0)
    return stamped
