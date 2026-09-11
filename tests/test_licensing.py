import numpy as np
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_origin
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import licensing

CRS = "EPSG:32719"
TRANSFORM = from_origin(0, 100, 10, 10)  # 10m pixels, origin at (0,100)
SHAPE = (10, 10)  # 100m x 100m grid


def _lease_gdf(minx, miny, maxx, maxy, situacion=None):
    geom = box(minx, miny, maxx, maxy)
    data = {"geometry": [geom]}
    if situacion is not None:
        data["situacion"] = [situacion]
    return gpd.GeoDataFrame(data, crs=CRS)


def test_rasterize_leases_empty_gdf_returns_all_false():
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS)
    mask = licensing.rasterize_leases(empty, SHAPE, TRANSFORM, CRS)
    assert mask.shape == SHAPE
    assert not mask.any()


def test_rasterize_leases_burns_polygon_correctly():
    # Polygon covering the left half of the grid (x: 0-50, y: 50-100 -> rows 0-4, cols 0-4).
    gdf = _lease_gdf(0, 50, 50, 100)
    mask = licensing.rasterize_leases(gdf, SHAPE, TRANSFORM, CRS)
    assert mask[0:5, 0:5].all()
    assert not mask[0:5, 5:10].any()


def test_classify_legal_status_three_tiers():
    new_mask = np.ones(SHAPE, dtype=bool)
    active_mask = np.zeros(SHAPE, dtype=bool)
    active_mask[0:3, 0:3] = True  # legal zone

    any_concession_mask = active_mask.copy()
    any_concession_mask[3:6, 3:6] = True  # disputed zone (concession, but not active)

    status = licensing.classify_legal_status(new_mask, active_mask, any_concession_mask)

    assert (status[0:3, 0:3] == licensing.LegalStatus.LEGAL).all()
    assert (status[3:6, 3:6] == licensing.LegalStatus.DISPUTED).all()
    assert (status[6:10, 6:10] == licensing.LegalStatus.ILLEGAL).all()


def test_classify_legal_status_respects_new_disturbance_mask():
    # Even inside a concession, a pixel that ISN'T new disturbance stays NOT_MINING.
    new_mask = np.zeros(SHAPE, dtype=bool)
    active_mask = np.ones(SHAPE, dtype=bool)
    any_concession_mask = np.ones(SHAPE, dtype=bool)

    status = licensing.classify_legal_status(new_mask, active_mask, any_concession_mask)
    assert (status == licensing.LegalStatus.NOT_MINING).all()


def test_status_confidence_discounts_disputed_relative_to_illegal():
    status = np.array([
        [licensing.LegalStatus.ILLEGAL, licensing.LegalStatus.DISPUTED],
        [licensing.LegalStatus.LEGAL, licensing.LegalStatus.NOT_MINING],
    ])
    proba = np.full((2, 2), 0.9)
    change_conf = np.full((2, 2), 0.9)

    conf = licensing.status_confidence(status, proba, change_conf)
    illegal_score = conf[0, 0]
    disputed_score = conf[0, 1]

    assert illegal_score > 0
    assert disputed_score > 0
    assert disputed_score < illegal_score  # disputed gets a discount
    assert conf[1, 0] == 0  # legal pixels never scored
    assert conf[1, 1] == 0  # not-mining pixels never scored


def test_suspicious_area_mask_excludes_licensed():
    new_mask = np.ones(SHAPE, dtype=bool)
    licensed = np.zeros(SHAPE, dtype=bool)
    licensed[0:5, :] = True
    suspicious = licensing.suspicious_area_mask(new_mask, licensed)
    assert not suspicious[0:5, :].any()
    assert suspicious[5:10, :].all()
