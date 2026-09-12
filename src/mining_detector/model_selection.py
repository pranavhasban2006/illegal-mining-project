"""Full tabular ML methodology for O2, matching the faculty-mandated workflow:

    Load dataset -> preprocessing -> data split (holdout N:N or 10-fold CV)
    -> SMOTE (on training folds only) -> train multiple algorithms
    -> overfit/underfit check + regularization -> test -> select best model

Operates on the same per-pixel feature table detection.py builds from
indices.compute_feature_stack -- this module IS the "CSV file" of the
flowchart, just materialized in memory instead of read from disk.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE

from . import indices
from .qc_smote import QCSMOTE


def _make_oversampler(variant: str, random_state: int):
    if variant == "standard":
        return SMOTE(random_state=random_state)
    if variant == "qc":
        return QCSMOTE(random_state=random_state)
    raise ValueError(f"Unknown smote_variant {variant!r}, expected 'standard' or 'qc'")

SPLIT_SCHEMES = {"80:20": 0.20, "70:30": 0.30, "60:40": 0.40}

# Overfit flag: train-score minus test-score beyond this gap.
OVERFIT_GAP_THRESHOLD = 0.10

# Cap dataset size for the multi-algorithm comparison (SVM in particular is
# O(n^2)-ish) -- the weak/hand labels are already a small labeled subset, but
# this keeps runtime bounded regardless of AOI size.
MAX_TRAINING_SAMPLES = 8000


def _candidate_models():
    """Algorithms compared, each with a regularization/early-stopping story."""
    return {
        "logistic_regression_l2": LogisticRegression(
            l1_ratio=0.0, C=1.0, max_iter=2000, class_weight="balanced",
            solver="saga",
        ),
        "logistic_regression_l1": LogisticRegression(
            l1_ratio=1.0, C=1.0, max_iter=2000, class_weight="balanced",
            solver="saga",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=0, n_jobs=-1,
        ),
        "svm_rbf": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, class_weight="balanced"), ensemble=False
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, validation_fraction=0.15,
            n_iter_no_change=10, tol=1e-4,  # early stopping
            random_state=0,
        ),
    }


def _build_dataset(bands: np.ndarray, labels: np.ndarray, terrain: dict | None = None):
    """labels: (H,W) with -1=unlabeled/excluded, 0=not-mining, 1=mining."""
    feats = indices.compute_feature_stack(bands, terrain=terrain)
    n_feat, h, w = feats.shape
    X_full = feats.reshape(n_feat, -1).T
    y_full = labels.reshape(-1)
    mask = y_full != -1
    return X_full[mask], y_full[mask]


def _subsample(X, y, rng, max_samples=MAX_TRAINING_SAMPLES):
    if len(y) <= max_samples:
        return X, y
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_pos = min(len(idx_pos), max_samples // 2)
    n_neg = min(len(idx_neg), max_samples - n_pos)
    keep = np.concatenate([
        rng.choice(idx_pos, n_pos, replace=False),
        rng.choice(idx_neg, n_neg, replace=False),
    ])
    rng.shuffle(keep)
    return X[keep], y[keep]


def _evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "iou": jaccard_score(y, y_pred, average="binary", zero_division=0),
    }


def run_holdout(bands, labels, split_scheme="70:30", use_smote=True, random_state=0,
                 terrain=None, smote_variant="standard"):
    """Single train/test split with SMOTE-on-train, multi-algorithm comparison.

    smote_variant: "standard" (imblearn SMOTE) or "qc" (QCSMOTE, see qc_smote.py --
    a documented reimplementation of Upman & Gowda 2026's quality-controlled
    oversampling, cited in Project Review 1 as this project's identified
    next step beyond standard SMOTE).
    """
    rng = np.random.default_rng(random_state)
    X, y = _build_dataset(bands, labels, terrain=terrain)
    X, y = _subsample(X, y, rng)

    test_size = SPLIT_SCHEMES[split_scheme]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    if use_smote and y_train.sum() >= 6:  # SMOTE needs a handful of minority samples
        X_train_s, y_train = _make_oversampler(smote_variant, random_state).fit_resample(X_train_s, y_train)

    results = {}
    for name, model in _candidate_models().items():
        model.fit(X_train_s, y_train)
        train_metrics = _evaluate(model, X_train_s, y_train)
        test_metrics = _evaluate(model, X_test_s, y_test)
        overfit_gap = train_metrics["f1"] - test_metrics["f1"]
        results[name] = {
            "model": model,
            "train": train_metrics,
            "test": test_metrics,
            "overfit_gap": overfit_gap,
            "overfit_flag": overfit_gap > OVERFIT_GAP_THRESHOLD,
        }

    best_name = max(results, key=lambda n: results[n]["test"]["f1"])
    return {
        "scaler": scaler,
        "split_scheme": split_scheme,
        "used_smote": use_smote,
        "smote_variant": smote_variant if use_smote else None,
        "results": results,
        "best_model_name": best_name,
        "best_model": results[best_name]["model"],
    }


def run_cross_validation(bands, labels, n_splits=10, use_smote=True, random_state=0, terrain=None,
                          smote_variant="standard"):
    """10-fold stratified CV, SMOTE applied inside each fold's training portion."""
    rng = np.random.default_rng(random_state)
    X, y = _build_dataset(bands, labels, terrain=terrain)
    X, y = _subsample(X, y, rng)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    candidates = _candidate_models()
    fold_scores = {name: [] for name in candidates}

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_s, X_val_s = scaler.transform(X_train), scaler.transform(X_val)

        if use_smote and y_train.sum() >= 6:
            X_train_s, y_train_fold = _make_oversampler(smote_variant, random_state).fit_resample(X_train_s, y_train)
        else:
            y_train_fold = y_train

        for name, template in candidates.items():
            model = clone(template)
            model.fit(X_train_s, y_train_fold)
            fold_scores[name].append(_evaluate(model, X_val_s, y_val)["f1"])

    mean_scores = {name: float(np.mean(s)) for name, s in fold_scores.items()}
    best_name = max(mean_scores, key=mean_scores.get)

    # Refit the best algorithm on the full (scaled, SMOTE'd) dataset for deployment.
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    if use_smote and y.sum() >= 6:
        X_s, y_fit = _make_oversampler(smote_variant, random_state).fit_resample(X_s, y)
    else:
        y_fit = y
    best_model = candidates[best_name]
    best_model.fit(X_s, y_fit)

    return {
        "scaler": scaler,
        "n_splits": n_splits,
        "used_smote": use_smote,
        "smote_variant": smote_variant if use_smote else None,
        "fold_scores": fold_scores,
        "mean_scores": mean_scores,
        "best_model_name": best_name,
        "best_model": best_model,
    }
