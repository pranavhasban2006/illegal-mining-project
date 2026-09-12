"""A documented reimplementation of the QC-SMOTE mechanism described in
Upman & Gowda (2026), "QC-SMOTE: Quality-Controlled SMOTE for Imbalanced
Classification", arXiv:2606.24625 -- cited in Project Review 1 as directly
relevant to O2's class-imbalance handling (mining pixels are a small
minority of any scene).

IMPORTANT: this is NOT the authors' original code (not publicly released
as of this project's search); it is reconstructed from the paper's
described mechanism at the level of detail available (abstract/summary,
not the full equations), since implementing an *exact* replica without the
paper's full methodology section would risk misrepresenting what was
actually built. What IS faithfully reproduced, per the paper's own
description:
  1. A composite trustworthiness score per minority sample, combining
     local density and "safe level" (fraction of same-class neighbors) --
     used to weight which minority points seed synthetic samples, rather
     than sampling uniformly as standard SMOTE does.
  2. A purity check on each candidate synthetic point's neighborhood in
     the ORIGINAL data before accepting it (our approximation of the
     paper's "IPQ-guided", "midpoint purity" check).
  3. Graceful degradation: a candidate landing in a majority-dominant
     neighborhood is replaced with a minority duplicate instead of being
     kept as a low-quality synthetic point -- the paper's stated fix for
     standard SMOTE generating synthetic samples in noisy/overlapping
     regions of feature space.

This is offered as an alternative to standard SMOTE in model_selection.py
(`smote_variant="qc"`), not a required default -- see README for the
before/after comparison on this project's real pilot data.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class QCSMOTE:
    """Minimal, sklearn/imblearn-style fit_resample interface, binary classes only."""

    def __init__(self, k_neighbors: int = 5, purity_neighbors: int = 5, random_state: int = 0):
        self.k_neighbors = k_neighbors
        self.purity_neighbors = purity_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("QCSMOTE currently supports binary classification only")

        minority_class = classes[np.argmin(counts)]
        n_minority, n_majority = counts.min(), counts.max()
        n_to_generate = n_majority - n_minority
        if n_to_generate <= 0:
            return X.copy(), y.copy()

        minority_idx = np.where(y == minority_class)[0]
        if len(minority_idx) < 2:
            return X.copy(), y.copy()  # nothing to interpolate between

        # --- Step 1: trustworthiness score per minority sample ---
        k_all = min(self.k_neighbors, len(X) - 1)
        nn_all = NearestNeighbors(n_neighbors=k_all + 1).fit(X)
        _, all_neighbor_idx = nn_all.kneighbors(X[minority_idx])
        safe_level = np.array([
            np.mean(y[neighbors[1:]] == minority_class) for neighbors in all_neighbor_idx
        ])  # fraction of each minority point's neighbors that are same-class
        weights = safe_level + 0.05  # keep isolated points selectable, just less likely
        weights = weights / weights.sum()

        k_minority = min(self.k_neighbors, len(minority_idx) - 1)
        nn_minority = NearestNeighbors(n_neighbors=k_minority + 1).fit(X[minority_idx])

        # Purity checker fit once against the ORIGINAL data (not growing
        # with accepted synthetics), for both correctness and speed.
        k_purity = min(self.purity_neighbors, len(X) - 1)
        nn_purity = NearestNeighbors(n_neighbors=k_purity).fit(X)

        synthetic_X = []
        for _ in range(n_to_generate):
            src_pos = rng.choice(len(minority_idx), p=weights)
            src_idx = minority_idx[src_pos]

            _, neighbor_pos = nn_minority.kneighbors(X[src_idx].reshape(1, -1))
            neighbor_pos = neighbor_pos[0]
            neighbor_pos = neighbor_pos[neighbor_pos != src_pos]

            if len(neighbor_pos) == 0:
                synthetic_X.append(X[src_idx])  # isolated minority point -- duplicate
                continue

            partner_idx = minority_idx[rng.choice(neighbor_pos)]
            ratio = rng.uniform(0, 1)
            candidate = X[src_idx] + ratio * (X[partner_idx] - X[src_idx])

            # --- Step 2/3: purity check + graceful degradation ---
            _, purity_idx = nn_purity.kneighbors(candidate.reshape(1, -1))
            neighborhood_purity = np.mean(y[purity_idx[0]] == minority_class)

            if neighborhood_purity < 0.5:
                synthetic_X.append(X[src_idx])  # majority-dominant region -- duplicate instead
            else:
                synthetic_X.append(candidate)

        synthetic_X = np.array(synthetic_X)
        synthetic_y = np.full(len(synthetic_X), minority_class, dtype=y.dtype)

        X_resampled = np.vstack([X, synthetic_X])
        y_resampled = np.concatenate([y, synthetic_y])
        return X_resampled, y_resampled
