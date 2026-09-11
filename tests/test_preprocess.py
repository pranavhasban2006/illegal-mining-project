import numpy as np
from rasterio.transform import from_origin
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import preprocess

# Identity-CRS setup (WGS84 transform == WGS84 target) so pixel math is exact
# without needing a real projected CRS round-trip.
CRS = "EPSG:4326"
PIXEL_DEG = 0.001
ORIGIN_LON, ORIGIN_LAT = -70.0, -12.0
TRANSFORM = from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG)


def test_crop_to_bbox_shape_matches_requested_extent():
    bands = np.arange(3 * 100 * 100, dtype=np.float32).reshape(3, 100, 100)
    # 20x20 pixel window starting 10 pixels in from the origin.
    minx = ORIGIN_LON + 10 * PIXEL_DEG
    maxx = ORIGIN_LON + 30 * PIXEL_DEG
    maxy = ORIGIN_LAT - 10 * PIXEL_DEG
    miny = ORIGIN_LAT - 30 * PIXEL_DEG

    cropped, new_transform = preprocess.crop_to_bbox(bands, TRANSFORM, CRS, (minx, miny, maxx, maxy))
    assert cropped.shape == (3, 20, 20)


def test_crop_to_bbox_preserves_pixel_values():
    bands = np.zeros((1, 50, 50), dtype=np.float32)
    bands[0, 15, 15] = 42.0  # a marker pixel

    minx = ORIGIN_LON + 10 * PIXEL_DEG
    maxx = ORIGIN_LON + 20 * PIXEL_DEG
    maxy = ORIGIN_LAT - 10 * PIXEL_DEG
    miny = ORIGIN_LAT - 20 * PIXEL_DEG

    cropped, _ = preprocess.crop_to_bbox(bands, TRANSFORM, CRS, (minx, miny, maxx, maxy))
    # Marker at (row=15, col=15) in the full array should land at (5, 5) in the crop.
    assert cropped[0, 5, 5] == 42.0


def test_crop_to_bbox_clamps_to_array_bounds():
    bands = np.zeros((1, 10, 10), dtype=np.float32)
    # A bbox that extends well past the array's actual extent.
    minx = ORIGIN_LON - 5 * PIXEL_DEG
    maxx = ORIGIN_LON + 50 * PIXEL_DEG
    maxy = ORIGIN_LAT + 5 * PIXEL_DEG
    miny = ORIGIN_LAT - 50 * PIXEL_DEG

    cropped, _ = preprocess.crop_to_bbox(bands, TRANSFORM, CRS, (minx, miny, maxx, maxy))
    assert cropped.shape[1] <= 10
    assert cropped.shape[2] <= 10
