"""Tests for model selection, per-scene normalization and the spatial split.

The data are synthetic: mining pixels are clustered in one part of the grid and
shifted in feature space, so a correct pipeline must reach a high test F1 and must not
leak information between train and test.
"""

from __future__ import annotations

import numpy as np
import pytest

from mining_detector.change_detection import normalize_per_scene
from mining_detector.detection import MiningDetector
from mining_detector import model_selection as ms
from mining_detector.model_selection import (
    fit_and_select,
    make_split,
    smote_train_only,
    spatial_block_ids,
    subsample,
)

FEATURES = [f"f{i}" for i in range(6)]


def _synthetic_labelled(seed: int = 0, n: int = 6000, grid: int = 400):
    """Rows in a grid. Positives sit in the top-right quadrant and are shifted in feature space."""
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, grid, size=n)
    cols = rng.integers(0, grid, size=n)
    mining_zone = (rows < 200) & (cols >= 200)
    y = (mining_zone & (rng.random(n) < 0.8)).astype(int)
    X = rng.normal(0.0, 1.0, size=(n, len(FEATURES)))
    X[y == 1] += 2.5
    return X, y, rows, cols


def test_spatial_split_keeps_blocks_whole_and_disjoint():
    X, y, rows, cols = _synthetic_labelled()
    block = spatial_block_ids(rows, cols, block_px=100)
    rng = np.random.default_rng(1)
    split = make_split(block, y, rng, kind="spatial_blocks")
    train_blocks = set(block[split.train].tolist())
    test_blocks = set(block[split.test].tolist())
    assert train_blocks.isdisjoint(test_blocks), "a block was split between train and test"
    assert len(split.train) + len(split.test) == len(y)
    assert len(split.test) > 0


def test_smote_runs_on_train_only_and_never_touches_test():
    X, y, rows, cols = _synthetic_labelled(seed=2)
    block = spatial_block_ids(rows, cols, block_px=100)
    split = make_split(block, y, np.random.default_rng(3), kind="spatial_blocks")
    X_tr, y_tr = X[split.train], y[split.train]
    X_te, y_te = X[split.test], y[split.test]

    X_bal, y_bal = smote_train_only(X_tr, y_tr, seed=3)
    # Synthetic rows are added to the training side only: the balanced set is larger, and
    # the test set we keep is exactly the original rows, untouched.
    assert len(y_bal) > len(y_tr)
    assert np.bincount(y_bal)[1] == np.bincount(y_bal)[0] or abs(
        np.bincount(y_bal)[1] - np.bincount(y_bal)[0]) <= 1
    assert X_te.shape == (len(split.test), len(FEATURES))
    assert np.array_equal(y_te, y[split.test])


def test_model_selection_finds_separable_signal():
    X, y, rows, cols = _synthetic_labelled(seed=4)
    sel = fit_and_select(X, y, rows, cols, FEATURES, seed=4, cap=6000, block_px=100)
    assert sel.best.f1_test > 0.9
    assert sel.best.f1_val > 0.9
    assert sel.split_kind == "spatial_blocks_three_way"
    names = [c.name for c in sel.candidates]
    assert {"logistic_l1", "logistic_l2", "random_forest", "svm_rbf", "gradient_boosting"} <= set(names)
    assert set(sel.feature_importance) == set(FEATURES)
    assert all(isinstance(c.overfit_flag, bool) for c in sel.candidates)


def test_model_selection_refuses_one_class():
    X, _, rows, cols = _synthetic_labelled(seed=5)
    with pytest.raises(ValueError, match="Not enough labelled examples"):
        fit_and_select(X, np.zeros(len(X), dtype=int), rows, cols, FEATURES)


def test_subsample_keeps_both_classes_under_cap():
    y = np.array([0] * 900 + [1] * 100)
    idx = subsample(y, cap=200, rng=np.random.default_rng(0))
    assert len(idx) <= 210
    assert set(np.unique(y[idx])) == {0, 1}


def test_per_scene_normalization_removes_global_shift_and_gain():
    """A scene that is a brightened, shifted copy of another should give the same probability map."""
    rng = np.random.default_rng(7)
    h, w = 60, 60
    stack = rng.normal(0.3, 0.1, size=(h, w, len(FEATURES))).astype(np.float32)
    stack[:20, :20] += 0.5   # a bright patch so the map is not flat
    valid = np.ones((h, w), dtype=bool)

    from sklearn.linear_model import LogisticRegression

    X_train = normalize_per_scene(stack, valid)[::5, ::5].reshape(-1, len(FEATURES))
    y_train = (X_train[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    det = MiningDetector(model, FEATURES, normalize_per_scene=True)

    original = det.predict_proba_map(stack, valid)
    shifted = stack * 2.5 + 0.7     # a different haze and exposure
    adjusted = det.predict_proba_map(shifted, valid)
    assert np.max(np.abs(original - adjusted)) < 1e-3


def test_detector_rejects_wrong_feature_count():
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression().fit(np.random.rand(20, 3), np.tile([0, 1], 10))
    det = MiningDetector(model, ["a", "b", "c"])
    with pytest.raises(ValueError, match="features"):
        det.predict_proba_map(np.zeros((4, 4, 5), dtype=np.float32), np.ones((4, 4), dtype=bool))


def test_test_split_is_never_used_during_selection(monkeypatch):
    """The test rows must not reach model selection, and are scored exactly once.

    select_on_validation is handed only train and validation arrays. score_test_once is the
    only function that sees the test rows, and it must run exactly once, on the chosen model.
    """
    X, y, rows, cols = _synthetic_labelled(seed=6)
    test_ids = set()
    selection_inputs = []
    scored = []

    real_select = ms.select_on_validation
    real_score = ms.score_test_once

    def spy_select(X_tr, y_tr, X_va, y_va, seed, notes):
        selection_inputs.extend([X_tr, X_va])
        return real_select(X_tr, y_tr, X_va, y_va, seed, notes)

    def spy_score(model, X_te, y_te):
        test_ids.add(id(X_te))
        scored.append(model)
        return real_score(model, X_te, y_te)

    monkeypatch.setattr(ms, "select_on_validation", spy_select)
    monkeypatch.setattr(ms, "score_test_once", spy_score)

    sel = fit_and_select(X, y, rows, cols, FEATURES, seed=6, cap=6000, block_px=100)

    assert len(scored) == 1, "test split must be scored exactly once"
    assert scored[0] is sel.best.model, "the single test score must be for the selected model"
    assert len(test_ids) == 1
    # The arrays given to selection must not be the test array.
    assert all(id(arr) not in test_ids for arr in selection_inputs)
    # select_on_validation has no test parameters at all.
    import inspect
    params = inspect.signature(real_select).parameters
    assert not any("test" in p for p in params)


def test_three_way_split_sets_are_block_disjoint():
    X, y, rows, cols = _synthetic_labelled(seed=8)
    block = spatial_block_ids(rows, cols, block_px=50)
    sp = ms.make_three_way_split(block, y, np.random.default_rng(8))
    blocks = [set(block[i].tolist()) for i in (sp.train, sp.val, sp.test)]
    assert blocks[0].isdisjoint(blocks[1]) and blocks[0].isdisjoint(blocks[2])
    assert blocks[1].isdisjoint(blocks[2])
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(y)
    for idx in (sp.train, sp.val, sp.test):
        assert len(np.unique(y[idx])) == 2
