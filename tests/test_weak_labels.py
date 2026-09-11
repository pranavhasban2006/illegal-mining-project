import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import weak_labels


def _bands_with_gradient(shape=(20, 20)):
    """A scene with a gradient from bare soil (top) to dense forest (bottom),
    so percentile thresholds have a real distribution to work with."""
    bands = np.zeros((6, *shape), dtype=np.float32)
    t = np.linspace(0, 1, shape[0])[:, None] * np.ones((1, shape[1]))  # 0=soil, 1=forest

    bands[0] = 0.2 * (1 - t) + 0.05 * t   # blue
    bands[1] = 0.22 * (1 - t) + 0.08 * t  # green
    bands[2] = 0.25 * (1 - t) + 0.05 * t  # red
    bands[3] = 0.15 * (1 - t) + 0.45 * t  # NIR (low for soil, high for forest)
    bands[4] = 0.32 * (1 - t) + 0.2 * t   # SWIR1
    bands[5] = 0.28 * (1 - t) + 0.1 * t   # SWIR2
    return bands


def test_generate_weak_labels_has_three_classes():
    bands = _bands_with_gradient()
    labels = weak_labels.generate_weak_labels(bands, positive_percentile=90, negative_percentile=90)
    unique = set(np.unique(labels))
    assert unique <= {weak_labels.UNLABELED, weak_labels.NOT_MINING, weak_labels.MINING}
    assert weak_labels.MINING in unique
    assert weak_labels.NOT_MINING in unique


def test_generate_weak_labels_positive_at_bare_end():
    bands = _bands_with_gradient()
    labels = weak_labels.generate_weak_labels(bands, positive_percentile=90, negative_percentile=90)
    # Top rows are the bare-soil end of the gradient.
    assert (labels[0, :] == weak_labels.MINING).any()
    # Bottom rows are the dense-forest end.
    assert (labels[-1, :] == weak_labels.NOT_MINING).any()


def test_label_summary_percentages_sum_correctly():
    labels = np.array([
        [weak_labels.MINING, weak_labels.NOT_MINING],
        [weak_labels.UNLABELED, weak_labels.UNLABELED],
    ])
    summary = weak_labels.label_summary(labels)
    assert summary["n_mining"] == 1
    assert summary["n_not_mining"] == 1
    assert summary["n_unlabeled"] == 2
    assert abs(summary["pct_labeled"] - 50.0) < 1e-9


def test_higher_positive_percentile_yields_fewer_positive_labels():
    bands = _bands_with_gradient()
    loose = weak_labels.generate_weak_labels(bands, positive_percentile=80, negative_percentile=80)
    strict = weak_labels.generate_weak_labels(bands, positive_percentile=98, negative_percentile=80)
    assert (strict == weak_labels.MINING).sum() <= (loose == weak_labels.MINING).sum()
