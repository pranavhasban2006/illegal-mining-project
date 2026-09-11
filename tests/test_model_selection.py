import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import model_selection


def _separable_bands_and_labels(shape=(30, 30), seed=0):
    """A synthetic scene with an obviously separable mining rectangle."""
    rng = np.random.default_rng(seed)
    bands = np.zeros((6, *shape), dtype=np.float32)
    # Vegetation background.
    bands[0] = rng.normal(0.05, 0.005, shape)
    bands[1] = rng.normal(0.08, 0.005, shape)
    bands[2] = rng.normal(0.05, 0.005, shape)
    bands[3] = rng.normal(0.45, 0.01, shape)
    bands[4] = rng.normal(0.2, 0.01, shape)
    bands[5] = rng.normal(0.1, 0.01, shape)

    labels = np.zeros(shape, dtype=np.int8)
    # Bare-soil mining rectangle, spectrally distinct.
    bands[0, 5:15, 5:15] = rng.normal(0.2, 0.01, (10, 10))
    bands[1, 5:15, 5:15] = rng.normal(0.22, 0.01, (10, 10))
    bands[2, 5:15, 5:15] = rng.normal(0.25, 0.01, (10, 10))
    bands[3, 5:15, 5:15] = rng.normal(0.15, 0.01, (10, 10))
    bands[4, 5:15, 5:15] = rng.normal(0.32, 0.01, (10, 10))
    bands[5, 5:15, 5:15] = rng.normal(0.28, 0.01, (10, 10))
    labels[5:15, 5:15] = 1

    return np.clip(bands, 0, 1), labels


def test_run_holdout_returns_expected_structure():
    bands, labels = _separable_bands_and_labels()
    report = model_selection.run_holdout(bands, labels, split_scheme="70:30", use_smote=True)

    assert report["split_scheme"] == "70:30"
    assert report["used_smote"] is True
    assert "best_model_name" in report
    assert report["best_model_name"] in report["results"]
    for name, res in report["results"].items():
        assert "train" in res and "test" in res
        assert 0.0 <= res["test"]["f1"] <= 1.0


def test_run_holdout_separates_easy_classes_well():
    bands, labels = _separable_bands_and_labels()
    report = model_selection.run_holdout(bands, labels, split_scheme="70:30", use_smote=True)
    best = report["results"][report["best_model_name"]]
    assert best["test"]["f1"] > 0.9  # trivially separable synthetic data


def test_run_holdout_without_smote_does_not_crash():
    bands, labels = _separable_bands_and_labels()
    report = model_selection.run_holdout(bands, labels, split_scheme="80:20", use_smote=False)
    assert report["used_smote"] is False
    assert report["best_model_name"] in report["results"]


def test_run_cross_validation_returns_mean_scores_for_every_candidate():
    bands, labels = _separable_bands_and_labels()
    report = model_selection.run_cross_validation(bands, labels, n_splits=3, use_smote=True)
    assert set(report["mean_scores"].keys()) == set(report["fold_scores"].keys())
    assert report["best_model_name"] in report["mean_scores"]
    for scores in report["fold_scores"].values():
        assert len(scores) == 3
