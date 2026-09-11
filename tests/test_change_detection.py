import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import change_detection


def test_new_disturbance_only_flags_pixels_crossing_threshold():
    proba_t0 = np.zeros((10, 10))
    proba_t1 = np.zeros((10, 10))

    # Was already mining at t0 -- should NOT count as "new".
    proba_t0[0:5, 0:5] = 0.9
    proba_t1[0:5, 0:5] = 0.95

    # Newly mining at t1, large enough blob to survive filtering.
    proba_t1[5:10, 5:10] = 0.9

    mask = change_detection.new_disturbance_mask(proba_t0, proba_t1, threshold=0.5, min_blob_pixels=5)
    assert mask[0:5, 0:5].sum() == 0
    assert mask[5:10, 5:10].sum() == 25


def test_small_blobs_filtered_as_noise():
    proba_t0 = np.zeros((20, 20))
    proba_t1 = np.zeros((20, 20))
    proba_t1[0, 0] = 0.9  # single-pixel "new" blob

    mask = change_detection.new_disturbance_mask(proba_t0, proba_t1, threshold=0.5, min_blob_pixels=10)
    assert mask.sum() == 0


def test_change_confidence_is_clipped_nonnegative():
    proba_t0 = np.full((5, 5), 0.8)
    proba_t1 = np.full((5, 5), 0.2)  # decrease -- should clip to 0, not go negative
    conf = change_detection.change_confidence(proba_t0, proba_t1)
    assert np.all(conf == 0)


def test_change_confidence_magnitude():
    proba_t0 = np.full((5, 5), 0.1)
    proba_t1 = np.full((5, 5), 0.9)
    conf = change_detection.change_confidence(proba_t0, proba_t1)
    np.testing.assert_allclose(conf, 0.8)


def test_no_change_gives_empty_mask():
    proba = np.random.default_rng(0).uniform(0, 1, (30, 30))
    mask = change_detection.new_disturbance_mask(proba, proba)
    assert mask.sum() == 0
