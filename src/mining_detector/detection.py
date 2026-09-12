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
from sklearn.preprocessing import StandardScaler

from . import indices, model_selection


class MiningDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selection_report = None
        self.feature_names = indices.FEATURE_NAMES
        self._fitted = False

    def fit(self, bands: np.ndarray, labels: np.ndarray, strategy: str = "70:30",
            use_smote: bool = True, cv_folds: int | None = None, terrain: dict | None = None,
            smote_variant: str = "standard"):
        """bands: (B,H,W) raw band stack.
        labels: (H,W) ground truth -- 1=mining, 0=not-mining, -1=unlabeled/excluded
        (weak_labels.py produces this format; a fully binary 0/1 mask also works).
        strategy: one of model_selection.SPLIT_SCHEMES ("80:20"/"70:30"/"60:40"),
        ignored if cv_folds is set.
        cv_folds: if given, runs stratified k-fold CV instead of a holdout split.
        terrain: optional {"elevation", "dist_water"} covariates (see terrain.py);
        if passed here, the SAME dict must be passed to predict_proba_map too.
        smote_variant: "standard" (imblearn SMOTE) or "qc" (see qc_smote.py).
        """
        if cv_folds:
            report = model_selection.run_cross_validation(
                bands, labels, n_splits=cv_folds, use_smote=use_smote, terrain=terrain,
                smote_variant=smote_variant,
            )
        else:
            report = model_selection.run_holdout(
                bands, labels, split_scheme=strategy, use_smote=use_smote, terrain=terrain,
                smote_variant=smote_variant,
            )

        self.selection_report = report
        self.model = report["best_model"]
        self.scaler = report["scaler"]
        self.feature_names = indices.FEATURE_NAMES + (indices.TERRAIN_FEATURE_NAMES if terrain else [])
        self._fitted = True
        return report

    def predict_proba_map(self, bands: np.ndarray, normalize_per_scene: bool = True,
                           terrain: dict | None = None) -> np.ndarray:
        """Returns (H,W) probability of mining-disturbed land.

        normalize_per_scene: re-fit the feature scaler on THIS scene's own
        statistics rather than reusing the training scene's scaler. This
        matters for bi-temporal comparison -- two Sentinel-2 scenes from
        different dates differ in atmosphere/sun-angle/season on top of any
        real land-cover change, so applying one scene's absolute scaling to
        another's raw reflectance miscalibrates the classifier and creates
        spurious "change" across nearly the whole image (see README). Each
        scene normalized against itself keeps the classifier looking at
        relative spectral position, which is far more comparable across dates.
        terrain: must match whatever was (or wasn't) passed to fit() -- pass
        the SAME static elevation/dist_water arrays for every date being
        compared (they don't change scene-to-scene), so this covariate
        doesn't itself introduce a spurious cross-date signal.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predicting.")
        feats = indices.compute_feature_stack(bands, terrain=terrain)
        n_feat, h, w = feats.shape
        X = feats.reshape(n_feat, -1).T
        scaler = StandardScaler().fit(X) if normalize_per_scene else self.scaler
        X = scaler.transform(X)
        proba = self.model.predict_proba(X)[:, 1]
        return proba.reshape(h, w)

    def predict_mask(self, bands: np.ndarray, threshold: float = 0.5, terrain: dict | None = None) -> np.ndarray:
        return (self.predict_proba_map(bands, terrain=terrain) >= threshold).astype(np.uint8)

    def feature_importances(self) -> dict | None:
        if not self._fitted:
            raise RuntimeError("Call fit() before inspecting importances.")
        if not hasattr(self.model, "feature_importances_"):
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))
