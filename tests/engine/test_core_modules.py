"""Unit tests for the A1 engine modules: config, indices, change detection, licensing, alerts.

Every test uses small synthetic arrays whose correct answer is known in advance.
No network access is needed.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from mining_detector.alerts import build_feature_collection, mask_to_polygons, write_geojson
from mining_detector.change_detection import (
    min_area_filter,
    new_disturbance_mask,
    normalize_per_scene,
    rule_mining_probability,
    spectral_disturbance_score,
)
from mining_detector.config import Config
from mining_detector.indices import FEATURE_NAMES, compute_feature_stack, ndvi
from mining_detector.licensing import classify_alerts

UTM = "EPSG:32719"


# ---------- config ----------

def test_config_defaults_are_valid():
    Config().validate()


def test_config_rejects_inverted_bbox():
    with pytest.raises(ValueError, match="Bounding box"):
        Config(bbox=(-69.4, -12.7, -69.6, -12.5)).validate()


def test_config_rejects_bad_thresholds():
    with pytest.raises(ValueError, match="p_low"):
        Config(p_low=0.8, p_high=0.5).validate()


# ---------- indices ----------

def test_ndvi_known_values():
    nir = np.array([[0.5]], dtype=np.float32)
    red = np.array([[0.1]], dtype=np.float32)
    assert ndvi(nir, red)[0, 0] == pytest.approx(0.4 / 0.6, rel=1e-4)


def test_feature_stack_shape_and_names():
    bands = {b: np.full((4, 5), 0.2, dtype=np.float32)
             for b in ("B02", "B03", "B04", "B08", "B11", "B12")}
    stack, names = compute_feature_stack(bands)
    assert stack.shape == (4, 5, len(FEATURE_NAMES)) == (4, 5, 10)
    assert names == list(FEATURE_NAMES)
    # Equal bands give zero NDVI everywhere.
    assert np.allclose(stack[..., names.index("NDVI")], 0.0, atol=1e-5)


def test_feature_stack_missing_band_is_clear():
    with pytest.raises(KeyError, match="B12"):
        compute_feature_stack({"B02": np.zeros((2, 2)), "B03": np.zeros((2, 2)),
                               "B04": np.zeros((2, 2)), "B08": np.zeros((2, 2)),
                               "B11": np.zeros((2, 2))})


# ---------- change detection ----------

def test_normalize_per_scene_ignores_invalid_pixels_and_centres_data():
    rng = np.random.default_rng(0)
    stack = rng.normal(5.0, 2.0, size=(50, 50, 3)).astype(np.float32)
    valid = np.ones((50, 50), dtype=bool)
    valid[:5] = False
    out = normalize_per_scene(stack, valid)
    assert np.allclose(out[valid].mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(out[valid].std(axis=0), 1.0, atol=1e-3)
    assert np.all(out[~valid] == 0)


def test_new_disturbance_mask_needs_low_before_and_high_after():
    p0 = np.array([[0.1, 0.9, 0.1]])
    p1 = np.array([[0.9, 0.9, 0.5]])
    assert new_disturbance_mask(p0, p1, p_high=0.7, p_low=0.3).tolist() == [[True, False, False]]


def test_spectral_score_flags_forest_to_bare_only():
    forest = np.zeros((1, 2, 10), dtype=np.float32)
    bare = np.zeros((1, 2, 10), dtype=np.float32)
    names = list(FEATURE_NAMES)
    forest[..., names.index("NDVI")] = 0.7
    forest[..., names.index("BSI")] = -0.2
    bare[..., names.index("NDVI")] = 0.05
    bare[..., names.index("BSI")] = 0.25
    score = spectral_disturbance_score(forest, bare)
    assert np.all(score == 1.0)
    assert np.all(spectral_disturbance_score(forest, forest) == 0.0)


def test_rule_probability_bare_high_forest_low():
    names = list(FEATURE_NAMES)
    stack = np.zeros((1, 2, 10), dtype=np.float32)
    stack[0, 0, names.index("NDVI")] = 0.05
    stack[0, 0, names.index("BSI")] = 0.2
    stack[0, 1, names.index("NDVI")] = 0.8
    stack[0, 1, names.index("BSI")] = -0.3
    p = rule_mining_probability(stack)
    assert p[0, 0] == pytest.approx(1.0)
    assert p[0, 1] == pytest.approx(0.0)


def test_min_area_filter_drops_small_patches_only():
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:4, 2:4] = True        # 4 px, tiny
    mask[10:30, 10:30] = True    # 400 px, big
    # 10 m pixels: 1 px = 100 m2 = 0.01 ha. Threshold 0.5 ha = 50 px.
    out = min_area_filter(mask, pixel_area_m2=100.0, min_area_ha=0.5)
    assert not out[2:4, 2:4].any()
    assert out[10:30, 10:30].all()


# ---------- licensing ----------

def _alerts_at(*boxes):
    return gpd.GeoDataFrame(geometry=list(boxes), crs=UTM)


def test_licensing_legal_disputed_illegal():
    concessions = gpd.GeoDataFrame(
        {"status": ["active", "expired"]},
        geometry=[box(0, 0, 100, 100), box(1000, 0, 1100, 100)],
        crs=UTM,
    )
    alerts = _alerts_at(
        box(10, 10, 50, 50),        # inside active concession -> legal
        box(1010, 10, 1050, 50),    # inside expired concession -> disputed
        box(300, 0, 350, 50),       # 200 m from active, inside 500 m buffer -> disputed
        box(5000, 5000, 5100, 5100) # far away -> illegal
    )
    labels, _ = classify_alerts(alerts, concessions, buffer_m=500.0)
    assert labels == ["legal", "disputed", "disputed", "illegal"]


def test_licensing_no_concessions_is_all_illegal():
    labels, notes = classify_alerts(_alerts_at(box(0, 0, 10, 10)), None)
    assert labels == ["illegal"]
    assert notes


def test_licensing_rejects_crs_mismatch():
    concessions = gpd.GeoDataFrame({"status": ["active"]}, geometry=[box(0, 0, 1, 1)],
                                   crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS"):
        classify_alerts(_alerts_at(box(0, 0, 1, 1)), concessions)


# ---------- alerts ----------

def test_mask_to_polygons_area_and_confidence():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True      # 100 px = 100 * 100 m2 = 1 ha
    prob = np.where(mask, 0.8, 0.0).astype(np.float32)
    t = Affine(10, 0, 500_000, 0, -10, 8_600_000)
    gdf = mask_to_polygons(mask, t, UTM, prob=prob)
    assert len(gdf) == 1
    assert gdf.area_ha.iloc[0] == pytest.approx(1.0)
    assert gdf.confidence.iloc[0] == pytest.approx(0.8)


def test_empty_alerts_give_valid_empty_collection():
    empty = mask_to_polygons(np.zeros((5, 5), dtype=bool), Affine(10, 0, 0, 0, -10, 0), UTM)
    fc = build_feature_collection(empty, [], "2025-07-27")
    assert fc == {"type": "FeatureCollection", "features": []}


def test_feature_collection_matches_contract(tmp_path):
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:15, 5:15] = True
    t = Affine(10, 0, 500_000, 0, -10, 8_600_000)
    gdf = mask_to_polygons(mask, t, UTM, prob=np.full(mask.shape, 0.9, dtype=np.float32))
    fc = build_feature_collection(gdf, ["illegal"], "2025-07-27")
    path = write_geojson(fc, tmp_path / "alerts.geojson")
    data = json.loads(path.read_text(encoding="utf-8"))
    props = data["features"][0]["properties"]
    assert set(props) == {"id", "legal_status", "confidence", "area_ha", "detected_on"}
    assert props["legal_status"] == "illegal"
    assert props["detected_on"] == "2025-07-27"
    # Coordinates must be longitude/latitude (Peru is around -69, -12).
    lon, lat = data["features"][0]["geometry"]["coordinates"][0][0]
    assert -71 < lon < -67 and -14 < lat < -10
