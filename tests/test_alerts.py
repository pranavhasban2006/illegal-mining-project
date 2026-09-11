import numpy as np
from rasterio.transform import from_origin
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import alerts, licensing

CRS = "EPSG:32719"
TRANSFORM = from_origin(0, 1000, 10, 10)  # 10m pixels -> each pixel = 100 sqm
SHAPE = (20, 20)


def test_vectorize_alerts_filters_small_polygons():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[0, 0] = 1  # 1 pixel = 100 sqm, below default min_area_m2=500
    conf = np.full(SHAPE, 0.8)

    gdf = alerts.vectorize_alerts(mask, conf, TRANSFORM, CRS, "2026-01-01")
    assert len(gdf) == 0


def test_vectorize_alerts_keeps_large_polygons_with_correct_confidence():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[0:10, 0:10] = 1  # 100 pixels = 10,000 sqm = 1 ha
    conf = np.zeros(SHAPE)
    conf[0:10, 0:10] = 0.75

    gdf = alerts.vectorize_alerts(mask, conf, TRANSFORM, CRS, "2026-01-01", min_area_m2=500)
    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert abs(row["area_ha"] - 1.0) < 0.01
    assert abs(row["confidence"] - 0.75) < 0.01
    assert row["alert_id"].startswith("ALERT-2026-01-01-")


def test_vectorize_status_alerts_separates_disputed_and_illegal():
    status = np.zeros(SHAPE, dtype=np.uint8)
    status[0:10, 0:10] = licensing.LegalStatus.ILLEGAL
    status[10:20, 10:20] = licensing.LegalStatus.DISPUTED
    conf = np.zeros(SHAPE)
    conf[0:10, 0:10] = 0.9
    conf[10:20, 10:20] = 0.6

    gdf = alerts.vectorize_status_alerts(status, conf, TRANSFORM, CRS, "2026-01-01", min_area_m2=500)
    assert set(gdf["legal_status"]) == {"illegal", "disputed"}
    assert len(gdf) == 2

    illegal_row = gdf[gdf["legal_status"] == "illegal"].iloc[0]
    disputed_row = gdf[gdf["legal_status"] == "disputed"].iloc[0]
    assert abs(illegal_row["confidence"] - 0.9) < 0.01
    assert abs(disputed_row["confidence"] - 0.6) < 0.01


def test_vectorize_status_alerts_never_includes_legal_pixels():
    status = np.full(SHAPE, licensing.LegalStatus.LEGAL, dtype=np.uint8)
    conf = np.full(SHAPE, 0.9)
    gdf = alerts.vectorize_status_alerts(status, conf, TRANSFORM, CRS, "2026-01-01")
    assert len(gdf) == 0


def test_empty_mask_returns_empty_geodataframe_with_expected_columns():
    mask = np.zeros(SHAPE, dtype=np.uint8)
    conf = np.zeros(SHAPE)
    gdf = alerts.vectorize_alerts(mask, conf, TRANSFORM, CRS, "2026-01-01")
    assert len(gdf) == 0
    assert "confidence" in gdf.columns
    assert "area_ha" in gdf.columns
