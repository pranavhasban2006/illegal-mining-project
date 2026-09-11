import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import indices


def _bands(blue, green, red, nir, swir1, swir2, shape=(3, 3)):
    """Builds a (6,H,W) band stack where every pixel has the given values."""
    vals = [blue, green, red, nir, swir1, swir2]
    return np.stack([np.full(shape, v, dtype=np.float32) for v in vals], axis=0)


def test_ndvi_high_for_vegetation():
    # Healthy vegetation: high NIR, low red.
    bands = _bands(blue=0.05, green=0.08, red=0.05, nir=0.45, swir1=0.2, swir2=0.1)
    result = indices.ndvi(bands)
    assert np.all(result > 0.6)


def test_ndvi_low_for_bare_soil():
    # Bare soil: red and NIR close together -> NDVI near zero.
    bands = _bands(blue=0.15, green=0.18, red=0.2, nir=0.22, swir1=0.3, swir2=0.25)
    result = indices.ndvi(bands)
    assert np.all(result < 0.2)


def test_ndwi_positive_for_water():
    # Water: green > NIR (NIR strongly absorbed).
    bands = _bands(blue=0.05, green=0.1, red=0.04, nir=0.02, swir1=0.01, swir2=0.01)
    result = indices.ndwi(bands)
    assert np.all(result > 0)


def test_bsi_high_for_bare_soil_vs_vegetation():
    soil = _bands(blue=0.15, green=0.18, red=0.22, nir=0.2, swir1=0.3, swir2=0.25)
    veg = _bands(blue=0.05, green=0.08, red=0.05, nir=0.45, swir1=0.2, swir2=0.1)
    assert indices.bsi(soil).mean() > indices.bsi(veg).mean()


def test_no_division_by_zero_nan_or_inf():
    bands = np.zeros((6, 4, 4), dtype=np.float32)
    for fn in (indices.ndvi, indices.ndmi, indices.bsi, indices.ndwi, indices.mining_disturbance_index):
        result = fn(bands)
        assert np.all(np.isfinite(result))


def test_compute_feature_stack_shape_and_order():
    bands = _bands(0.1, 0.1, 0.1, 0.3, 0.2, 0.15, shape=(5, 7))
    feats = indices.compute_feature_stack(bands)
    assert feats.shape == (len(indices.FEATURE_NAMES), 5, 7)
    # First 6 features are the raw bands, unchanged.
    np.testing.assert_array_equal(feats[0], bands[0])
    np.testing.assert_array_equal(feats[5], bands[5])


def test_mining_disturbance_index_clipped_to_range():
    bands = _bands(blue=0.9, green=0.9, red=0.9, nir=0.01, swir1=0.9, swir2=0.9)
    result = indices.mining_disturbance_index(bands)
    assert np.all(result <= 1.0) and np.all(result >= -1.0)
