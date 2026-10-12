"""Read, crop, resample, mask and cache Sentinel-2 bands for an area of interest.

Key design choices:
- Only the window that covers the area is read from each file (no full-tile download).
- Both dates are read onto ONE target grid computed from the area, so the two dates
  are co-registered pixel for pixel.
- Bands at 20 m (B11, B12, SCL) are resampled to 10 m; reflectance bands use bilinear
  resampling and the SCL class map uses nearest-neighbour so its codes are not blended.
- Cloud, shadow, cirrus and saturated pixels are masked using the SCL band.
- Results are cached in data/cache/ keyed by scene id and the area, so reruns are fast.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from .config import S2_BANDS

log = logging.getLogger(__name__)

# Sentinel-2 Scene Classification Layer codes that make a pixel unusable.
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 medium cloud, 9 high cloud, 10 thin cirrus.
SCL_INVALID = (0, 1, 3, 8, 9, 10)
SCL_WATER = 6

# Some assets are named by band (B02) on Planetary Computer and by colour on Earth Search.
ASSET_ALIASES = {
    "B02": ("B02", "blue"),
    "B03": ("B03", "green"),
    "B04": ("B04", "red"),
    "B08": ("B08", "nir"),
    "B11": ("B11", "swir16"),
    "B12": ("B12", "swir22"),
    "SCL": ("SCL", "scl"),
}

_GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.jp2",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
)


@dataclass
class Grid:
    """A target grid: transform, shape (rows, cols) and CRS."""

    transform: Affine
    shape: tuple[int, int]
    epsg: int

    @property
    def crs(self) -> str:
        return f"EPSG:{self.epsg}"

    @property
    def pixel_area_m2(self) -> float:
        return abs(self.transform.a * self.transform.e)


@dataclass
class SceneData:
    """One date, cropped to the area and resampled to the target grid."""

    bands: dict[str, np.ndarray]     # reflectance 0-1, float32, keyed by band name
    scl: np.ndarray                  # uint8 class map
    valid: np.ndarray                # bool, True where the pixel is usable
    grid: Grid
    item_id: str = ""
    acquired: str = ""
    cloud_pct: float = float("nan")
    source: str = ""
    baseline: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean()) if self.valid.size else 0.0


def utm_epsg_for_bbox(bbox: tuple[float, float, float, float]) -> int:
    """Return the UTM EPSG code for the centre of the bounding box (WGS84 datum)."""
    w, s, e, n = bbox
    lon = (w + e) / 2.0
    lat = (s + n) / 2.0
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def target_grid(bbox: tuple[float, float, float, float], res: float = 10.0,
                epsg: int | None = None) -> Grid:
    """Build the 10 m grid that covers the bounding box, snapped to whole pixels."""
    from rasterio.warp import transform_bounds as tb

    epsg = epsg or utm_epsg_for_bbox(bbox)
    left, bottom, right, top = tb("EPSG:4326", f"EPSG:{epsg}", *bbox)
    left = np.floor(left / res) * res
    bottom = np.floor(bottom / res) * res
    right = np.ceil(right / res) * res
    top = np.ceil(top / res) * res
    width = int(round((right - left) / res))
    height = int(round((top - bottom) / res))
    transform = Affine(res, 0, left, 0, -res, top)
    return Grid(transform=transform, shape=(height, width), epsg=epsg)


def scene_cache_key(item_id: str, bbox, res: float, epsg: int) -> str:
    """A short, stable file name for one scene and area."""
    raw = f"{item_id}|{tuple(round(v, 6) for v in bbox)}|{res}|{epsg}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def reflectance_offset(baseline: str) -> int:
    """Processing baseline 04.00 and later added +1000 to DN values; subtract it back out."""
    try:
        major, minor = baseline.split(".")[:2]
        return -1000 if (int(major), int(minor)) >= (4, 0) else 0
    except (ValueError, AttributeError):
        return 0  # unknown baseline: assume the older, offset-free convention


def dn_to_reflectance(dn: np.ndarray, baseline: str, boa_offset_applied: bool = False) -> np.ndarray:
    """Convert Sentinel-2 digital numbers to surface reflectance in the range 0-1.

    Args:
        dn: raw band values.
        baseline: processing baseline string, used only when the offset is not already applied.
        boa_offset_applied: True when the catalogue has already removed the +1000 offset
            (Earth Search sets this as earthsearch:boa_offset_applied). Subtracting it again
            would push visible bands to zero, so it is skipped in that case.
    """
    off = 0 if boa_offset_applied else reflectance_offset(baseline)
    refl = (dn.astype(np.float32) + off) / 10_000.0
    return np.clip(refl, 0.0, 1.0).astype(np.float32)


def valid_mask(scl: np.ndarray, mask_water: bool = False) -> np.ndarray:
    """True where the SCL class is usable. Optionally also exclude water pixels."""
    bad = np.isin(scl, SCL_INVALID)
    if mask_water:
        bad |= scl == SCL_WATER
    return ~bad


def _read_to_grid(href: str, grid: Grid, resampling: Resampling) -> np.ndarray:
    """Read the part of one raster that covers the grid, reprojected onto the grid."""
    with rasterio.Env(**_GDAL_ENV), rasterio.open(href) as src:
        src_bounds = transform_bounds(grid.crs, src.crs, *_grid_bounds(grid))
        win = from_bounds(*src_bounds, transform=src.transform).round_offsets().round_lengths()
        pad = 2  # a little padding so edge pixels resample correctly
        win = Window(win.col_off - pad, win.row_off - pad, win.width + 2 * pad, win.height + 2 * pad)
        data = src.read(1, window=win, boundless=True, fill_value=0)
        win_transform = src.window_transform(win)
        dst = np.zeros(grid.shape, dtype=data.dtype)
        reproject(
            source=data,
            destination=dst,
            src_transform=win_transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=resampling,
        )
    return dst


def _grid_bounds(grid: Grid) -> tuple[float, float, float, float]:
    h, w = grid.shape
    t = grid.transform
    return (t.c, t.f + h * t.e, t.c + w * t.a, t.f)  # left, bottom, right, top


def load_scene_from_hrefs(
    hrefs: dict[str, str],
    grid: Grid,
    baseline: str = "",
    mask_water: bool = False,
    meta: dict | None = None,
    boa_offset_applied: bool = False,
) -> SceneData:
    """Read the bands for one date from a dict of file locations (local paths or URLs).

    Args:
        hrefs: keys are band names ("B02"..."B12", "SCL"), values are file paths or URLs.
        grid: the shared target grid.
        baseline: Sentinel-2 processing baseline, for the reflectance offset.
        mask_water: also mark water pixels as invalid.
        meta: optional dict with item_id, acquired, cloud_pct and source.
        boa_offset_applied: True if the catalogue already removed the +1000 offset.

    Raises:
        KeyError: if a required band is missing from hrefs.
    """
    missing = [b for b in (*S2_BANDS, "SCL") if b not in hrefs]
    if missing:
        raise KeyError(f"Scene is missing bands: {', '.join(missing)}")

    bands = {}
    for name in S2_BANDS:
        dn = _read_to_grid(hrefs[name], grid, Resampling.bilinear)
        bands[name] = dn_to_reflectance(dn, baseline, boa_offset_applied)
    scl = _read_to_grid(hrefs["SCL"], grid, Resampling.nearest).astype(np.uint8)
    valid = valid_mask(scl, mask_water=mask_water)

    meta = meta or {}
    scene = SceneData(
        bands=bands, scl=scl, valid=valid, grid=grid,
        item_id=meta.get("item_id", ""), acquired=meta.get("acquired", ""),
        cloud_pct=float(meta.get("cloud_pct", float("nan"))),
        source=meta.get("source", ""), baseline=baseline,
    )
    if scene.valid_fraction < 0.05:
        scene.notes.append(
            f"Only {scene.valid_fraction:.0%} of the area is clear on this date; "
            "results for it will be unreliable."
        )
    return scene


def _hrefs_from_item(item) -> dict[str, str]:
    """Pick the file location for each band from a STAC item, handling either asset naming."""
    out = {}
    for band, aliases in ASSET_ALIASES.items():
        for key in aliases:
            if key in item.assets:
                out[band] = item.assets[key].href
                break
    return out


def load_scene(choice, bbox, grid: Grid | None = None, cache_dir: str | Path | None = None,
               mask_water: bool = False) -> SceneData:
    """Load one SceneChoice for an area, using the cache when possible.

    Args:
        choice: SceneChoice from data_sources.search_scenes.
        bbox: area of interest (west, south, east, north).
        grid: optional pre-computed shared grid. Computed from bbox when omitted.
        cache_dir: folder for cached .npz files. No caching when None.
        mask_water: also mask water pixels.
    """
    grid = grid or target_grid(bbox)
    meta = {"item_id": choice.item_id, "acquired": choice.acquired.isoformat(),
            "cloud_pct": choice.cloud_pct, "source": choice.source}

    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_cache_key(choice.item_id, bbox, 10.0, grid.epsg)
        cache_file = cache_dir / f"{key}.npz"
        if cache_file.exists():
            log.info("Using cached scene %s", cache_file.name)
            return _load_npz(cache_file, grid, mask_water)

    hrefs = _hrefs_from_item(choice.item)
    scene = load_scene_from_hrefs(hrefs, grid, baseline=choice.baseline,
                                  mask_water=mask_water, meta=meta,
                                  boa_offset_applied=choice.boa_offset_applied)
    if cache_file is not None:
        _save_npz(cache_file, scene)
    return scene


def _save_npz(path: Path, scene: SceneData) -> None:
    """Store reflectance bands (as uint16 to save space), the SCL map and metadata."""
    arrays = {f"b_{k}": np.round(v * 10_000).astype(np.uint16) for k, v in scene.bands.items()}
    arrays["scl"] = scene.scl
    meta = {
        "transform": list(scene.grid.transform)[:6],
        "shape": list(scene.grid.shape), "epsg": scene.grid.epsg,
        "item_id": scene.item_id, "acquired": scene.acquired,
        "cloud_pct": scene.cloud_pct, "source": scene.source, "baseline": scene.baseline,
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _load_npz(path: Path, grid: Grid, mask_water: bool) -> SceneData:
    """Load a cached scene. The cache holds 10 m UTM data, so the grid must match."""
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        bands = {k[2:]: z[k].astype(np.float32) / 10_000.0 for k in z.files if k.startswith("b_")}
        scl = z["scl"]
    cached_grid = Grid(Affine(*meta["transform"]), tuple(meta["shape"]), meta["epsg"])
    if cached_grid.shape != grid.shape or cached_grid.epsg != grid.epsg:
        raise RuntimeError(
            f"Cached scene {path.name} covers a different grid than the request. "
            "Delete data/cache/ and try again."
        )
    return SceneData(
        bands=bands, scl=scl, valid=valid_mask(scl, mask_water=mask_water), grid=cached_grid,
        item_id=meta["item_id"], acquired=meta["acquired"], cloud_pct=meta["cloud_pct"],
        source=meta["source"], baseline=meta["baseline"],
    )
