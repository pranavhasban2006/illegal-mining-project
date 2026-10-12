"""Choose the classifier honestly: a three-way spatial split, SMOTE on train only.

Why spatial blocks: neighbouring pixels look almost identical. A random pixel split
puts near-copies of every training pixel into the test set, which inflates the score.
Instead the area is cut into square blocks and whole blocks go to one side only.

Three sets, all made of whole blocks:
    train       - fits the candidate models (SMOTE is applied here and only here).
    validation  - used for model selection and the overfit check.
    test        - scored ONCE, for the model that validation selected. It is never used
                  to compare or choose models.

Steps:
1. Build a table of labelled pixels (features X, labels y, block id for each pixel).
2. Subsample to a cap (for speed) while keeping both classes.
3. Split by block: about 60% train, 20% validation, 20% test.
4. Oversample the minority class with SMOTE on the TRAIN rows only.
5. Fit each candidate on train. Measure F1 on train and on validation. Flag overfitting when
   train F1 minus validation F1 is above 0.05.
6. Pick the candidate with the best validation F1. Score that one model on the test set,
   once, and report that number as the headline. Return feature importances (random forest).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

log = logging.getLogger(__name__)

OVERFIT_GAP = 0.05
SVM_MAX_ROWS = 2_000     # the RBF SVM cost grows with its support vectors, so it trains on less
TRAIN_SCORE_MAX = 20_000 # train F1 (for the overfit check) is measured on a fixed subsample
DEFAULT_CAP = 60_000     # labelled rows used across train, validation and test


@dataclass
class Split:
    """Row indices for train and test (two-way split, used by the tests and helpers)."""

    train: np.ndarray
    test: np.ndarray
    kind: str  # "spatial_blocks" or "random_pixels"


@dataclass
class ThreeWaySplit:
    """Row indices for train, validation and test. The three sets never share a block."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class Candidate:
    """One fitted model and its scores. f1_test is None until the model is scored on test."""

    name: str
    model: object
    f1_train: float
    f1_val: float
    precision_val: float
    recall_val: float
    overfit_flag: bool
    fit_seconds: float = 0.0
    f1_test: float | None = None
    precision_test: float | None = None
    recall_test: float | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "fit_seconds": round(self.fit_seconds, 1),
            "f1_train": round(self.f1_train, 4),
            "f1_validation": round(self.f1_val, 4),
            "f1_test": None if self.f1_test is None else round(self.f1_test, 4),
            "precision_validation": round(self.precision_val, 4),
            "recall_validation": round(self.recall_val, 4),
            "overfit_flag": bool(self.overfit_flag),
        }


@dataclass
class Selection:
    """Result of model selection. best.f1_test is the headline, scored once on test."""

    best: Candidate
    candidates: list[Candidate]
    feature_importance: dict[str, float]
    split_kind: str
    notes: list[str] = field(default_factory=list)


def subsample(y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Return row indices of at most `cap` rows, keeping the class ratio roughly the same."""
    n = len(y)
    if n <= cap:
        return np.arange(n)
    idx = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        take = max(1, int(round(cap * len(cls_idx) / n)))
        idx.append(rng.choice(cls_idx, size=min(take, len(cls_idx)), replace=False))
    return np.sort(np.concatenate(idx))


def spatial_block_ids(rows: np.ndarray, cols: np.ndarray, block_px: int) -> np.ndarray:
    """Give every pixel the id of the square block it falls in."""
    n_block_cols = int(cols.max() // block_px) + 1
    return (rows // block_px) * n_block_cols + (cols // block_px)


def _both_classes(y: np.ndarray, idx: np.ndarray) -> bool:
    return len(np.unique(y[idx])) == 2


def make_three_way_split(
    block_id: np.ndarray, y: np.ndarray, rng: np.random.Generator,
    val_fraction: float = 0.2, test_fraction: float = 0.2, max_tries: int = 200,
) -> ThreeWaySplit:
    """Split blocks into train, validation and test, each containing both classes.

    Raises:
        ValueError: if no block arrangement puts both classes on all three sides.
    """
    blocks = np.unique(block_id)
    n_test = max(1, int(round(len(blocks) * test_fraction)))
    n_val = max(1, int(round(len(blocks) * val_fraction)))
    if n_test + n_val >= len(blocks):
        raise ValueError("Too few spatial blocks for a train, validation and test split.")
    for _ in range(max_tries):
        rng.shuffle(blocks)
        is_test = np.isin(block_id, blocks[:n_test])
        is_val = np.isin(block_id, blocks[n_test:n_test + n_val])
        is_train = ~(is_test | is_val)
        train_idx, val_idx, test_idx = (np.flatnonzero(m) for m in (is_train, is_val, is_test))
        if all(_both_classes(y, i) for i in (train_idx, val_idx, test_idx)):
            return ThreeWaySplit(train=train_idx, val=val_idx, test=test_idx)
    raise ValueError(
        "Could not split the labelled area into spatial blocks that each contain mining and "
        "non-mining examples. Use a larger area."
    )


def make_split(
    block_id: np.ndarray, y: np.ndarray, rng: np.random.Generator, test_fraction: float = 0.2,
    kind: str = "spatial_blocks", max_tries: int = 50,
) -> Split:
    """Two-way split (train and test). Kept for helpers and tests; selection uses the three-way split.

    For spatial blocks, the draw is repeated until both classes appear on both sides.
    If that never happens, the function raises instead of silently switching to a random
    pixel split, because a random split would leak neighbouring pixels.
    """
    if kind == "random_pixels":
        perm = rng.permutation(len(y))
        cut = int(len(y) * (1 - test_fraction))
        return Split(train=np.sort(perm[:cut]), test=np.sort(perm[cut:]), kind=kind)

    blocks = np.unique(block_id)
    n_test = max(1, int(round(len(blocks) * test_fraction)))
    for _ in range(max_tries):
        rng.shuffle(blocks)
        is_test = np.isin(block_id, blocks[:n_test])
        train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
        if _both_classes(y, train_idx) and _both_classes(y, test_idx):
            return Split(train=train_idx, test=test_idx, kind=kind)
    raise ValueError(
        "Could not split the labelled area into spatial blocks that both contain mining and "
        "non-mining examples. Use a larger area, or pass a smaller block size."
    )


def smote_train_only(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class with SMOTE. Called on training rows only."""
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(y_train)
    if counts.min() < 6:
        log.warning("Too few minority rows for SMOTE; training without oversampling.")
        return X_train, y_train
    k = min(5, counts.min() - 1)
    return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X_train, y_train)


def _candidates(seed: int) -> dict[str, object]:
    """The five candidate models. Scaling is inside each pipeline, fitted on train rows only."""
    return {
        "logistic_l1": make_pipeline(StandardScaler(), LogisticRegression(
            l1_ratio=1.0, solver="saga", C=1.0, max_iter=2000, random_state=seed)),
        "logistic_l2": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)),
        # Histogram-based boosting with built-in early stopping: same idea as gradient boosting,
        # but much faster on large tables.
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_depth=6, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, random_state=seed),
    }


def _scores(model, X, y) -> tuple[float, float, float]:
    pred = model.predict(X)
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


def select_on_validation(
    X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
    seed: int, notes: list[str],
) -> tuple[list[Candidate], dict[str, object]]:
    """Fit every candidate on train and score it on validation.

    This function receives only the train and validation arrays. The test set is not
    available here, so it cannot influence which model is chosen.

    Returns:
        (candidates, fitted_models_by_name). Candidates have f1_val and the overfit flag.
    """
    rng = np.random.default_rng(seed)
    X_bal, y_bal = smote_train_only(X_tr, y_tr, seed)  # validation rows are never resampled
    svm_idx = subsample(y_bal, SVM_MAX_ROWS, rng)
    tr_score_idx = subsample(y_tr, TRAIN_SCORE_MAX, np.random.default_rng(seed + 1))
    notes.append(
        f"Train rows {len(y_tr)} (after SMOTE {len(y_bal)}), validation rows {len(y_va)}."
    )

    candidates: list[Candidate] = []
    fitted: dict[str, object] = {}
    for name, model in _candidates(seed).items():
        Xfit, yfit = (X_bal[svm_idx], y_bal[svm_idx]) if name == "svm_rbf" else (X_bal, y_bal)
        log.info("Fitting %s on %d rows", name, len(yfit))
        t_fit = time.time()
        model.fit(Xfit, yfit)
        fit_seconds = time.time() - t_fit
        f1_tr, _, _ = _scores(model, X_tr[tr_score_idx], y_tr[tr_score_idx])
        f1_va, p_va, r_va = _scores(model, X_va, y_va)
        candidates.append(Candidate(
            name=name, model=model, f1_train=f1_tr, f1_val=f1_va,
            precision_val=p_va, recall_val=r_va,
            overfit_flag=(f1_tr - f1_va) > OVERFIT_GAP, fit_seconds=fit_seconds,
        ))
        fitted[name] = model
        log.info("%s: F1 train %.3f validation %.3f (%.1f s)", name, f1_tr, f1_va, fit_seconds)
    return candidates, fitted


def score_test_once(model, X_te: np.ndarray, y_te: np.ndarray) -> tuple[float, float, float]:
    """Score the selected model on the test set. Call this exactly once per run."""
    return _scores(model, X_te, y_te)


def fit_and_select(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
    cap: int = DEFAULT_CAP,
    block_px: int = 100,
) -> Selection:
    """Run the full selection and return the best model with all the scores.

    Args:
        X: (N, F) feature matrix for labelled pixels only (already per-scene normalized).
        y: (N,) labels, 1 mining, 0 not mining.
        rows, cols: pixel coordinates of each row, used to form spatial blocks.
        feature_names: names for the F columns.
        seed: random seed.
        cap: maximum rows used across train, validation and test.
        block_px: side of the spatial block in pixels (100 px = 1 km at 10 m).

    Raises:
        ValueError: if there are too few positive or negative labels to learn from.
    """
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 50:
        raise ValueError(
            "Not enough labelled examples of both classes to train a model "
            f"(counts: {np.bincount(y).tolist()}). Try a larger area or add hand labels."
        )

    rng = np.random.default_rng(seed)
    keep = subsample(y, cap, rng)
    X, y, rows, cols = X[keep], y[keep], rows[keep], cols[keep]

    block = spatial_block_ids(rows, cols, block_px)
    split = make_three_way_split(block, y, rng)
    X_tr, y_tr = X[split.train], y[split.train]
    X_va, y_va = X[split.val], y[split.val]
    X_te, y_te = X[split.test], y[split.test]

    notes = [
        f"Three-way spatial split: train {len(y_tr)} rows, validation {len(y_va)}, "
        f"test {len(y_te)} (test positives {int(y_te.sum())}).",
        "Model selection used only train and validation. The test set was scored once, "
        "for the selected model.",
    ]

    candidates, fitted = select_on_validation(X_tr, y_tr, X_va, y_va, seed, notes)
    best = max(candidates, key=lambda c: c.f1_val)
    if best.overfit_flag:
        notes.append(
            f"The selected model ({best.name}) overfits: train F1 is "
            f"{best.f1_train - best.f1_val:.2f} above validation F1."
        )

    f1_te, p_te, r_te = score_test_once(best.model, X_te, y_te)
    best.f1_test, best.precision_test, best.recall_test = f1_te, p_te, r_te

    importance = {}
    for name, val in zip(feature_names, fitted["random_forest"].feature_importances_):
        importance[name] = round(float(val), 4)
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    return Selection(best=best, candidates=candidates, feature_importance=importance,
                     split_kind="spatial_blocks_three_way", notes=notes)
