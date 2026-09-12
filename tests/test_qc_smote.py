import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector.qc_smote import QCSMOTE


def _imbalanced_dataset(n_majority=100, n_minority=10, seed=0):
    rng = np.random.default_rng(seed)
    X_majority = rng.normal(0, 1, (n_majority, 4))
    X_minority = rng.normal(5, 0.5, (n_minority, 4))  # well-separated cluster
    X = np.vstack([X_majority, X_minority])
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


def test_qc_smote_balances_classes():
    X, y = _imbalanced_dataset()
    X_res, y_res = QCSMOTE(random_state=0).fit_resample(X, y)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]


def test_qc_smote_only_adds_minority_samples():
    X, y = _imbalanced_dataset()
    X_res, y_res = QCSMOTE(random_state=0).fit_resample(X, y)
    n_majority_before = (y == 0).sum()
    n_majority_after = (y_res == 0).sum()
    assert n_majority_before == n_majority_after  # majority class untouched


def test_qc_smote_noop_when_already_balanced():
    X, y = _imbalanced_dataset(n_majority=20, n_minority=20)
    X_res, y_res = QCSMOTE(random_state=0).fit_resample(X, y)
    assert len(y_res) == len(y)


def test_qc_smote_synthetic_points_land_near_minority_cluster():
    X, y = _imbalanced_dataset(n_majority=100, n_minority=10)
    X_res, y_res = QCSMOTE(random_state=0).fit_resample(X, y)
    synthetic = X_res[len(X):]
    # The minority cluster is centered at 5; synthetic points (from a
    # well-separated cluster) should land in that neighborhood, not
    # scattered near the majority cluster at 0.
    assert np.all(synthetic.mean(axis=1) > 2.5)


def test_qc_smote_rejects_non_binary():
    X = np.random.rand(30, 3)
    y = np.array([0] * 10 + [1] * 10 + [2] * 10)
    import pytest
    with pytest.raises(ValueError):
        QCSMOTE().fit_resample(X, y)


def test_qc_smote_handles_single_minority_sample_gracefully():
    X, y = _imbalanced_dataset(n_majority=20, n_minority=1)
    # Only 1 minority sample -- can't interpolate, should return unchanged
    # rather than crashing.
    X_res, y_res = QCSMOTE(random_state=0).fit_resample(X, y)
    assert len(y_res) == len(y)
