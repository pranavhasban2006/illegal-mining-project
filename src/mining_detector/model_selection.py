"""Choose the classifier honestly: spatial train/test split, SMOTE on train only.

Why spatial blocks: neighbouring pixels look almost identical. A random pixel split
puts near-copies of every training pixel into the test set, which inflates the score.
Instead the area is cut into square blocks and whole blocks go to either train or test.

Steps:
1. Build a table of labelled pixels (features X, labels y, block id for each pixel).
2. Subsample to a cap (for speed) while keeping both classes.
3. Split by block (80% of blocks train, 20% test). A random pixel split is also reported
   when cheap, for comparison only.
4. Oversample the minority class with SMOTE on the TRAIN rows only.
5. Fit each candidate, measure precision, recall and F1 on train and test, and flag overfitting
   when train F1 minus test F1 is above 0.05.
6. Pick the candidate with the best test F1 and return feature importances (random forest).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

log = logging.getLogger(__name__)

OVERFIT_GAP = 0.05
SVM_MAX_ROWS = 20_000
DEFAULT_CAP = 200_000


@dataclass
class Split:
    """Row indices for train and test, plus the split type used."""

    train: np.ndarray
    test: np.ndarray
    kind: str  # "spatial_blocks" or "random_pixels"


@dataclass
class Candidate:
    """One fitted model and its scores."""

    name: str
    model: object
    f1_train: float
    f1_test: float
    precision_test: float
    recall_test: float
    overfit_flag: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "f1_train": round(self.f1_train, 4),
            "f1_test": round(self.f1_test, 4),
            "precision_test": round(self.precision_test, 4),
            "recall_test": round(self.recall_test, 4),
            "overfit_flag": bool(self.overfit_flag),
        }


@dataclass
class Selection:
    """Result of model selection."""

    best: Candidate
    candidates: list[Candidate]
    feature_importance: dict[str, float]
    split_kind: str
    random_split_f1: dict[str, float] = field(default_factory=dict)
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


def make_split(
    block_id: np.ndarray, y: np.ndarray, rng: np.random.Generator, test_fraction: float = 0.2,
    kind: str = "spatial_blocks", max_tries: int = 50,
) -> Split:
    """Split rows into train and test. Blocks are kept whole, so no block straddles the split.

    For spatial blocks, the draw is repeated (up to max_tries) until both classes appear on
    both sides. If that never happens, the function raises instead of silently switching
    to a random pixel split, because a random split would leak neighbouring pixels.
    """
    if kind == "random_pixels":
        perm = rng.permutation(len(y))
        cut = int(len(y) * (1 - test_fraction))
        return Split(train=np.sort(perm[:cut]), test=np.sort(perm[cut:]), kind=kind)

    blocks = np.unique(block_id)
    n_test = max(1, int(round(len(blocks) * test_fraction)))
    for _ in range(max_tries):
        rng.shuffle(blocks)
        test_blocks = set(blocks[:n_test].tolist())
        is_test = np.isin(block_id, list(test_blocks))
        if len(np.unique(y[is_test])) == 2 and len(np.unique(y[~is_test])) == 2:
            return Split(train=np.flatnonzero(~is_test), test=np.flatnonzero(is_test), kind=kind)
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
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=3, validation_fraction=0.1,
            n_iter_no_change=10, random_state=seed),
    }


def _scores(model, X, y) -> tuple[float, float, float]:
    pred = model.predict(X)
    return (f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            recall_score(y, pred, zero_division=0))


def fit_and_select(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
    cap: int = DEFAULT_CAP,
    block_px: int = 100,
    compare_random_split: bool = True,
) -> Selection:
    """Run the full selection and return the best model with all the scores.

    Args:
        X: (N, F) feature matrix for labelled pixels only (already per-scene normalized).
        y: (N,) labels, 1 mining, 0 not mining.
        rows, cols: pixel coordinates of each row, used to form spatial blocks.
        feature_names: names for the F columns.
        seed: random seed.
        cap: maximum rows used for training and evaluation.
        block_px: side of the spatial block in pixels (100 px = 1 km at 10 m).
        compare_random_split: also score the best model on a random pixel split, for comparison.

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
    split = make_split(block, y, rng, kind="spatial_blocks")

    X_tr, y_tr = X[split.train], y[split.train]
    X_te, y_te = X[split.test], y[split.test]
    X_tr_bal, y_tr_bal = smote_train_only(X_tr, y_tr, seed)  # test rows are never resampled

    notes = [
        f"Split: {split.kind}; train rows {len(y_tr)} (after SMOTE {len(y_tr_bal)}), "
        f"test rows {len(y_te)}, test positives {int(y_te.sum())}.",
    ]
    svm_idx = subsample(y_tr_bal, SVM_MAX_ROWS, rng)

    candidates: list[Candidate] = []
    fitted = {}
    for name, model in _candidates(seed).items():
        Xfit, yfit = (X_tr_bal[svm_idx], y_tr_bal[svm_idx]) if name == "svm_rbf" else (X_tr_bal, y_tr_bal)
        log.info("Fitting %s on %d rows", name, len(yfit))
        model.fit(Xfit, yfit)
        f1_tr, _, _ = _scores(model, X_tr, y_tr)
        f1_te, p_te, r_te = _scores(model, X_te, y_te)
        gap = f1_tr - f1_te
        cand = Candidate(name=name, model=model, f1_train=f1_tr, f1_test=f1_te,
                         precision_test=p_te, recall_test=r_te,
                         overfit_flag=gap > OVERFIT_GAP)
        candidates.append(cand)
        fitted[name] = model
        log.info("%s: F1 train %.3f test %.3f", name, f1_tr, f1_te)

    best = max(candidates, key=lambda c: c.f1_test)
    if best.overfit_flag:
        notes.append(
            f"The best model ({best.name}) overfits: train F1 is {best.f1_train - best.f1_test:.2f} "
            "above test F1."
        )

    importance = {}
    rf = fitted["random_forest"]
    for name, val in zip(feature_names, rf.feature_importances_):
        importance[name] = round(float(val), 4)
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    random_scores = {}
    if compare_random_split and split.kind == "spatial_blocks":
        rsplit = make_split(block, y, rng, kind="random_pixels")
        Xr_tr, yr_tr = X[rsplit.train], y[rsplit.train]
        Xr_te, yr_te = X[rsplit.test], y[rsplit.test]
        bal_X, bal_y = smote_train_only(Xr_tr, yr_tr, seed)
        clone = _candidates(seed)[best.name]
        clone.fit(bal_X, bal_y)
        random_scores = {"best_model_f1_test_random_pixels": round(_scores(clone, Xr_te, yr_te)[0], 4)}
        notes.append(
            "A random pixel split gives a higher score for the same model; the gap shows how much "
            "neighbouring pixels inflate random splits."
        )

    return Selection(best=best, candidates=candidates, feature_importance=importance,
                     split_kind=split.kind, random_split_f1=random_scores, notes=notes)
