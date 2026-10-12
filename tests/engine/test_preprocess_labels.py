"""Tests for masking, grid alignment, caching and label building.

All inputs are small local GeoTIFFs or arrays made in the test, so no network is used.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import box

from mining_detector.data_sources import SceneChoice, _best
from mining_detector.labels import FETCH_HINT, build_labels
from mining_detector.preprocess import (
    Grid,
    _load_npz,
    _save_npz,
    dn_to_reflectance,
    load_scene_from_hrefs,
    reflectance_offset,
    scene_cache_key,
    target_grid,
    utm_epsg_for_bbox,
    valid_mask,
)

BBOX = (-69.52, -12.66, -69.50, -12.64)  # ~2.2 km square near the default example area


def _write_tif(path: Path, arr: np.ndarray, transform: Affine, epsg: int, dtype="uint16") -> str:
    h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=1, dtype=dtype,
        crs=CRS.from_epsg(epsg), transform=transform,
    ) as dst:
        dst.write(arr.astype(dtype), 1)
    return str(path)


# ---------- masking and reflectance ----------

def test_utm_zone_for_default_area_is_19s():
    assert utm_epsg_for_bbox(BBOX) == 32719


def test_valid_mask_drops_cloud_shadow_and_cirrus_keeps_vegetation():
    scl = np.array([[4, 9, 3, 8, 10, 6, 0, 1]], dtype=np.uint8)
    ok = valid_mask(scl)
    assert ok.tolist() == [[True, False, False, False, False, True, False, False]]
    # Optionally drop water too.
    assert valid_mask(scl, mask_water=True)[0, 5] == False  # noqa: E712


def test_baseline_offset_applies_from_04_00_only():
    assert reflectance_offset("05.10") == -1000
    assert reflectance_offset("04.00") == -1000
    assert reflectance_offset("02.14") == 0
    assert reflectance_offset("") == 0


def test_already_offset_corrected_data_is_not_offset_twice():
    """Earth Search sets earthsearch:boa_offset_applied; subtracting 1000 again zeroes visible bands."""
    dn = np.array([[210, 2710]], dtype=np.uint16)     # real Earth Search values for forest
    refl = dn_to_reflectance(dn, "05.11", boa_offset_applied=True)
    assert refl[0, 0] == pytest.approx(0.021)
    assert refl[0, 1] == pytest.approx(0.271)


def test_dn_to_reflectance_known_values():
    dn = np.array([[1100, 500]], dtype=np.uint16)
    refl = dn_to_reflectance(dn, "05.00")
    assert refl[0, 0] == pytest.approx(0.01)
    assert refl[0, 1] == 0.0  # clipped, not negative
    assert dn_to_reflectance(dn, "02.14")[0, 0] == pytest.approx(0.11)


# ---------- grid and co-registration ----------

def test_target_grid_covers_bbox_at_10m():
    grid = target_grid(BBOX)
    assert grid.epsg == 32719
    assert grid.transform.a == pytest.approx(10.0)
    h, w = grid.shape
    # 0.02 degrees of longitude at this latitude is about 2.2 km.
    assert 200 <= w <= 230 and 200 <= h <= 230


def test_reading_onto_shared_grid_is_repeatable(tmp_path):
    """Reading the same files twice onto the shared grid gives identical arrays."""
    grid = target_grid(BBOX)
    h, w = grid.shape
    rng = np.random.default_rng(0)
    field = rng.integers(200, 3000, size=(h, w)).astype(np.uint16)
    scl = np.full((h, w), 4, dtype=np.uint8)

    hrefs = {}
    for band in ("B02", "B03", "B04", "B08", "B11", "B12"):
        hrefs[band] = _write_tif(tmp_path / f"{band}.tif", field, grid.transform, grid.epsg)
    hrefs["SCL"] = _write_tif(tmp_path / "SCL.tif", scl, grid.transform, grid.epsg, dtype="uint8")

    s0 = load_scene_from_hrefs(hrefs, grid, baseline="05.10")
    s1 = load_scene_from_hrefs(hrefs, grid, baseline="05.10")
    assert s0.bands["B04"].shape == grid.shape
    assert np.array_equal(s0.bands["B04"], s1.bands["B04"])
    assert np.array_equal(s0.bands["B04"], s0.bands["B08"])  # same field in every band
    # Round trip: DN 3000 with baseline offset -1000 gives 0.2.
    assert s0.bands["B04"].max() == pytest.approx((field.max() - 1000) / 10000, rel=1e-4)


def test_load_scene_flags_mostly_cloudy_dates(tmp_path):
    grid = target_grid(BBOX)
    h, w = grid.shape
    band = np.full((h, w), 1000, dtype=np.uint16)
    scl = np.full((h, w), 9, dtype=np.uint8)  # all high cloud
    hrefs = {b: _write_tif(tmp_path / f"{b}.tif", band, grid.transform, grid.epsg)
             for b in ("B02", "B03", "B04", "B08", "B11", "B12")}
    hrefs["SCL"] = _write_tif(tmp_path / "scl.tif", scl, grid.transform, grid.epsg, dtype="uint8")
    scene = load_scene_from_hrefs(hrefs, grid)
    assert scene.valid_fraction == 0.0
    assert any("unreliable" in n for n in scene.notes)


def test_load_scene_missing_band_is_clear(tmp_path):
    grid = target_grid(BBOX)
    with pytest.raises(KeyError, match="B12"):
        load_scene_from_hrefs({"B02": "x"}, grid)


# ---------- cache ----------

def test_cache_key_is_stable_and_area_specific():
    k1 = scene_cache_key("S2A_X", BBOX, 10.0, 32719)
    assert k1 == scene_cache_key("S2A_X", BBOX, 10.0, 32719)
    assert k1 != scene_cache_key("S2A_X", (-69.6, -12.7, -69.5, -12.6), 10.0, 32719)


def test_cache_round_trip_keeps_mask_and_grid(tmp_path):
    grid = target_grid(BBOX)
    h, w = grid.shape
    rng = np.random.default_rng(1)
    scl = rng.choice([4, 9], size=(h, w)).astype(np.uint8)
    from mining_detector.preprocess import SceneData

    scene = SceneData(
        bands={"B04": rng.random((h, w)).astype(np.float32)},
        scl=scl, valid=valid_mask(scl), grid=grid, item_id="X", acquired="2025-07-27",
        cloud_pct=3.0, source="test", baseline="05.10",
    )
    path = tmp_path / "scene.npz"
    _save_npz(path, scene)
    back = _load_npz(path, grid, mask_water=False)
    assert np.array_equal(back.valid, scene.valid)
    assert back.grid.shape == grid.shape
    assert back.item_id == "X"


# ---------- scene choice ----------

def test_best_scene_is_least_cloudy_then_closest_to_date():
    class FakeItem:
        def __init__(self, cloud, day):
            self.properties = {"eo:cloud_cover": cloud}
            self.datetime = __import__("datetime").datetime(2025, 7, day)

    items = [FakeItem(5.0, 1), FakeItem(2.0, 20), FakeItem(2.0, 27)]
    best = _best(items, date(2025, 7, 27))
    assert best.properties["eo:cloud_cover"] == 2.0
    assert best.datetime.day == 27


def test_scene_choice_describe_mentions_cloud_and_source():
    choice = SceneChoice(item=None, source="earth_search", item_id="S2B_T",
                         acquired=date(2025, 7, 31), cloud_pct=0.03, epsg=32719, baseline="05.10")
    text = choice.describe()
    assert "0.0% cloud" in text and "earth_search" in text


# ---------- labels ----------

def _grid_and_ndvi(bbox=BBOX):
    grid = target_grid(bbox)
    ndvi = np.full(grid.shape, 0.7, dtype=np.float32)   # dense vegetation everywhere
    ndwi = np.zeros(grid.shape, dtype=np.float32)
    return grid, ndvi, ndwi


def _mining_gpkg(path: Path, bbox=BBOX) -> Path:
    """A 300 m square of mining, placed in the middle of the area."""
    w, s, e, n = bbox
    cx, cy = (w + e) / 2, (s + n) / 2
    poly = box(cx - 0.002, cy - 0.0015, cx + 0.002, cy + 0.0015)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326").to_file(path, driver="GPKG")
    return path


def test_labels_positives_negatives_and_buffer(tmp_path):
    grid, ndvi, ndwi = _grid_and_ndvi()
    _mining_gpkg(tmp_path / "global_mining" / "mining_polygons.gpkg")
    labels = build_labels(grid, ndvi, ndwi, BBOX, external_dir=tmp_path)
    assert labels.source == "global_mining_footprint"
    assert labels.n_pos > 0 and labels.n_neg > 0
    # No negative pixel may sit within 300 m (30 px) of a positive pixel.
    from scipy import ndimage
    pos = labels.y == 1
    dist_px = ndimage.distance_transform_edt(~pos)
    assert not ((labels.y == 0) & (dist_px * 10 <= 300)).any()


def test_labels_water_counts_as_negative(tmp_path):
    grid, ndvi, ndwi = _grid_and_ndvi()
    ndwi[:20, :20] = 0.5           # water in one corner
    ndvi[:20, :20] = -0.2
    _mining_gpkg(tmp_path / "global_mining" / "mining_polygons.gpkg")
    labels = build_labels(grid, ndvi, ndwi, BBOX, external_dir=tmp_path)
    assert labels.y[:20, :20].max() == 0
    assert labels.counts["negative_water_px"] == 400


def test_missing_labels_gives_fetch_instruction(tmp_path):
    grid, ndvi, ndwi = _grid_and_ndvi()
    with pytest.raises(FileNotFoundError, match="fetch_datasets.py mining"):
        build_labels(grid, ndvi, ndwi, BBOX, external_dir=tmp_path)
    assert "fetch_datasets.py mining" in FETCH_HINT


def test_concession_fallback_is_flagged(tmp_path):
    grid, ndvi, ndwi = _grid_and_ndvi()
    w, s, e, n = BBOX
    conc = tmp_path / "conc.gpkg"
    gpd.GeoDataFrame(geometry=[box(w, s, e, n)], crs="EPSG:4326").to_file(conc, driver="GPKG")
    labels = build_labels(grid, ndvi, ndwi, BBOX, external_dir=tmp_path, concessions_path=conc)
    assert labels.source == "concessions_fallback"
    assert any("weak" in note for note in labels.notes)


def test_hand_labels_override_automatic_labels(tmp_path):
    grid, ndvi, ndwi = _grid_and_ndvi()
    _mining_gpkg(tmp_path / "global_mining" / "mining_polygons.gpkg")
    base = build_labels(grid, ndvi, ndwi, BBOX, external_dir=tmp_path)
    # Pick a negative pixel and say, by hand, that it IS mining.
    neg_r, neg_c = np.argwhere(base.y == 0)[0]
    x = grid.transform.c + (neg_c + 0.5) * grid.transform.a
    y = grid.transform.f + (neg_r + 0.5) * grid.transform.e
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(x, y)
    hand = tmp_path / "hand.csv"
    hand.write_text(f"lon,lat,label\n{lon},{lat},1\n", encoding="utf-8")
    labels = build_labels(grid, ndvi, ndwi, BBOX, external_dir=tmp_path,
                          extra_labels_path=hand, neighborhood_px=1)
    assert labels.y[neg_r, neg_c] == 1
    assert labels.counts["hand_labels_px"] == 1
