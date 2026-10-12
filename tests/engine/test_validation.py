"""Tests for checking the mask against hand labels, and for the review-chip export."""

from __future__ import annotations

import csv
import json

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

from mining_detector.indices import FEATURE_NAMES, compute_feature_stack
from mining_detector.validation import (
    CHIP_CATEGORIES,
    evaluate_points,
    load_check_points,
    make_review_chips,
    write_validation_into_report,
)

UTM = "EPSG:32719"
TRANSFORM = Affine(10, 0, 500_000, 0, -10, 8_600_000)


def _ll_point(row: int, col: int, label) -> dict:
    """A labelled point at the centre of pixel (row, col), in lon/lat."""
    x = TRANSFORM.c + (col + 0.5) * TRANSFORM.a
    y = TRANSFORM.f + (row + 0.5) * TRANSFORM.e
    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(x, y)
    return {"lon": lon, "lat": lat, "label": label}


def test_evaluate_points_known_confusion_matrix():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    rows = [
        _ll_point(15, 15, 1),   # predicted mining, labelled mining    -> TP
        _ll_point(12, 12, 0),   # predicted mining, labelled not      -> FP
        _ll_point(40, 40, 0),   # not predicted, not mining           -> TN
        _ll_point(40, 41, 1),   # not predicted, mining               -> FN
        _ll_point(5, 5, 1),     # FN
        _ll_point(1000, 1000, 1),  # outside the grid, not scored
    ]
    import pandas as pd
    df = pd.DataFrame(rows)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    res = evaluate_points(pts, mask, TRANSFORM, UTM)
    assert res["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 2}
    assert res["n_outside_grid"] == 1
    assert res["precision"] == pytest.approx(0.5)
    assert res["recall"] == pytest.approx(1 / 3, rel=1e-3)


def test_load_check_points_drops_unsure(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("lon,lat,label\n-69.5,-12.6,1\n-69.4,-12.5,unsure\n-69.3,-12.4,0\n",
                    encoding="utf-8")
    gdf = load_check_points(path)
    assert len(gdf) == 2
    assert gdf.attrs["dropped_unsure"] == 1
    assert set(gdf["label"]) == {0, 1}


def test_load_check_points_needs_label_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("lon,lat\n-69.5,-12.6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="label"):
        load_check_points(path)


def test_validation_is_written_into_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"n_alerts": 0}), encoding="utf-8")
    write_validation_into_report(report, {"f1": 0.5})
    assert json.loads(report.read_text(encoding="utf-8"))["validation"] == {"f1": 0.5}


def _synthetic_stack(h: int = 300, w: int = 300):
    """Forest everywhere, with one bare block that the mask covers."""
    bands = {
        "B02": np.full((h, w), 0.03, np.float32), "B03": np.full((h, w), 0.06, np.float32),
        "B04": np.full((h, w), 0.04, np.float32), "B08": np.full((h, w), 0.35, np.float32),
        "B11": np.full((h, w), 0.15, np.float32), "B12": np.full((h, w), 0.08, np.float32),
    }
    bands["B04"][100:140, 100:140] = 0.25
    bands["B08"][100:140, 100:140] = 0.28
    bands["B11"][100:140, 100:140] = 0.35
    bands["B02"][100:140, 100:140] = 0.12
    stack, names = compute_feature_stack(bands)
    mask = np.zeros((h, w), dtype=bool)
    mask[100:140, 100:140] = True
    prob = np.where(mask, 0.9, 0.05).astype(np.float32)
    return stack, names, mask, prob


def test_review_chips_are_exported_with_template(tmp_path):
    stack, names, mask, prob = _synthetic_stack()
    csv_path = make_review_chips(tmp_path, mask, prob, stack, names, TRANSFORM, UTM, n=8, chip_px=32)
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert 1 <= len(rows) <= 8
    assert {r["category"] for r in rows} <= set(CHIP_CATEGORIES)
    for r in rows:
        assert (tmp_path / "chips" / r["file"]).exists()
        assert -71 < float(r["lon"]) < -67 and -14 < float(r["lat"]) < -10
        assert r["label"] == ""  # the template is blank for people to fill in


def test_review_chip_categories_are_ordered_by_name():
    assert CHIP_CATEGORIES == ("near_boundary", "uncertain_probability",
                               "bare_far_from_mine", "mid_ndvi")
    assert "NDVI" in FEATURE_NAMES
