import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import terrain


def test_dem_tile_keys_covers_bbox_crossing_degree_boundaries():
    # Matches the real pilot AOI: crosses both the -13 latitude and
    # -70 longitude integer-degree boundaries, needing a 2x2 tile grid.
    bbox = (-70.05, -13.05, -69.75, -12.80)
    keys = terrain._dem_tile_keys(bbox)
    assert len(keys) == 4
    assert any("S14_00_W071_00" in k for k in keys)
    assert any("S14_00_W070_00" in k for k in keys)
    assert any("S13_00_W071_00" in k for k in keys)
    assert any("S13_00_W070_00" in k for k in keys)


def test_dem_tile_keys_single_tile_when_bbox_fits_in_one():
    bbox = (-70.5, -13.5, -70.2, -13.2)
    keys = terrain._dem_tile_keys(bbox)
    assert len(keys) == 1
    assert "S14_00_W071_00" in keys[0]


def _bands_with_water_strip(shape=(20, 20)):
    """A scene with a water strip down the middle (NDWI > 0)."""
    bands = np.zeros((6, *shape), dtype=np.float32)
    bands[0] = 0.05  # blue
    bands[1] = 0.08  # green
    bands[2] = 0.05  # red
    bands[3] = 0.4   # NIR -- dry land baseline
    bands[4] = 0.2
    bands[5] = 0.1

    mid = shape[1] // 2
    bands[3, :, mid - 1:mid + 1] = 0.02  # NIR drops sharply over water -> NDWI > 0
    return bands


def test_compute_distance_to_water_zero_at_water_pixels():
    bands = _bands_with_water_strip()
    dist = terrain.compute_distance_to_water(bands, pixel_size_m=10.0)
    mid = bands.shape[2] // 2
    assert dist[10, mid] == 0.0


def test_compute_distance_to_water_increases_away_from_water():
    bands = _bands_with_water_strip()
    dist = terrain.compute_distance_to_water(bands, pixel_size_m=10.0)
    mid = bands.shape[2] // 2
    assert dist[10, 0] > dist[10, mid - 2]


def test_compute_distance_to_water_no_water_returns_finite_cap():
    bands = np.zeros((6, 10, 10), dtype=np.float32)
    bands[3] = 0.4  # high NIR everywhere -> NDWI negative everywhere, no water
    bands[1] = 0.05
    dist = terrain.compute_distance_to_water(bands, pixel_size_m=10.0)
    assert np.all(np.isfinite(dist))  # never inf -- would break a scaler/classifier
    assert np.all(dist > 0)
