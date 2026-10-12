"""Apply a trained classifier to a whole scene and get a mining-probability map.

Per-scene normalization: every scene is standardized with its OWN mean and standard
deviation before the classifier sees it (the same idea as re-fitting a StandardScaler
per scene). Without this, a hazier or differently lit image would shift every pixel and
look like new mining. Training rows are normalized the same way, so the model learns
on the same scale it is applied to.
"""

from __future__ import annotations

import numpy as np

from .change_detection import normalize_per_scene


class MiningDetector:
    """A fitted classifier plus the rules for turning a scene into a probability map.

    Args:
        model: any fitted scikit-learn classifier with predict_proba.
        feature_names: names of the feature columns, in order.
        normalize_per_scene: standardize each scene on its own before predicting.
    """

    def __init__(self, model, feature_names: list[str], normalize_per_scene: bool = True):
        if not hasattr(model, "predict_proba"):
            raise TypeError("The model must provide predict_proba.")
        self.model = model
        self.feature_names = list(feature_names)
        self.normalize_per_scene = normalize_per_scene

    def prepare(self, stack: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Return the stack as the model should see it (normalized when enabled)."""
        if stack.shape[-1] != len(self.feature_names):
            raise ValueError(
                f"Scene has {stack.shape[-1]} features but the model expects {len(self.feature_names)}."
            )
        if self.normalize_per_scene:
            return normalize_per_scene(stack, valid)
        return stack

    def predict_proba_map(self, stack: np.ndarray, valid: np.ndarray, batch: int = 500_000) -> np.ndarray:
        """Probability (0-1) of mining for every pixel. Invalid pixels get 0.

        Args:
            stack: (H, W, F) raw feature stack for one date.
            valid: (H, W) boolean mask of usable pixels.
            batch: number of pixels passed to the model at a time, to limit memory.
        """
        prepared = self.prepare(stack, valid)
        flat = prepared[valid]
        out = np.zeros(stack.shape[:2], dtype=np.float32)
        if flat.size == 0:
            return out
        probs = np.empty(len(flat), dtype=np.float32)
        for start in range(0, len(flat), batch):
            chunk = flat[start:start + batch]
            probs[start:start + batch] = self.model.predict_proba(chunk)[:, 1]
        out[valid] = probs
        return out
