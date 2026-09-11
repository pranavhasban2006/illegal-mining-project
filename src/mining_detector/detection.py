"""O2: Mining-area detection via per-pixel spectral-feature classification.

Follows the full tabular-ML methodology (see model_selection.py): data split
(holdout or 10-fold CV) -> SMOTE on training data -> train & compare several
algorithms -> overfit/underfit diagnostics -> select the best-scoring model.
A per-pixel classifier is used instead of a segmentation CNN because it
needs far fewer labeled pixels, trains in seconds, and stays explainable via
feature importances -- appropriate for a hackathon timeline. See indices.py
for the feature stack this consumes.
"""

from __future__ import annotations

import numpy as np

from . import indices, model_selection


class MiningDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selection_report = None
        self._fitted = False

    def fit(self, bands: np.ndarray, labels: np.ndarray, strategy: str = "70:30",
            use_smote: bool = True, cv_folds: int | None = None):
        """bands: (B,H,W) raw band stack.
        labels: (H,W) ground truth -- 1=mining, 0=not-mining, -1=unlabeled/excluded
        (weak_labels.py produces this format; a fully binary 0/1 mask also works).
        strategy: one of model_selection.SPLIT_SCHEMES ("80:20"/"70:30"/"60:40"),
        ignored if cv_folds is set.
        cv_folds: if given, runs stratified k-fold CV instead of a holdout split.
        """
        if cv_folds:
            report = model_selection.run_cross_validation(
                bands, labels, n_splits=cv_folds, use_smote=use_smote
            )
        else:
            report = model_selection.run_holdout(
                bands, labels, split_scheme=strategy, use_smote=use_smote
            )

        self.selection_report = report
        self.model = report["best_model"]
        self.scaler = report["scaler"]
        self._fitted = True
        return report

    def predict_proba_map(self, bands: np.ndarray) -> np.ndarray:
        """Returns (H,W) probability of mining-disturbed land."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predicting.")
        feats = indices.compute_feature_stack(bands)
        n_feat, h, w = feats.shape
        X = feats.reshape(n_feat, -1).T
        X = self.scaler.transform(X)
        proba = self.model.predict_proba(X)[:, 1]
        return proba.reshape(h, w)

    def predict_mask(self, bands: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba_map(bands) >= threshold).astype(np.uint8)

    def feature_importances(self) -> dict | None:
        if not self._fitted:
            raise RuntimeError("Call fit() before inspecting importances.")
        if not hasattr(self.model, "feature_importances_"):
            return None
        return dict(zip(indices.FEATURE_NAMES, self.model.feature_importances_))
