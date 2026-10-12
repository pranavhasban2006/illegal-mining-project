"""Folds human-labeled hard cases (data/hard_cases/) back into training as
real hard-negative/positive examples, with a proper held-out split so the
same chips are never used for both training and validation.

This is the direct fix for the validation gap found in
data/hard_cases/validation_summary.md: the classifier over-flags moderate
vegetation loss (fallow/agriculture/regrowth) as mining. Concession-based
labels (real_labels.py) never included examples of that specific confusion,
so the model had nothing to learn the distinction from. These hand labels
do.
"""

from __future__ import annotations

import numpy as np

MINING = 1
NOT_MINING = 0


def single_labeler_votes(votes: dict) -> dict:
    """Keeps only the flat {chip_id: label} entries.

    Votes saved by the dashboard use {"chip_id__labeler": {...}} keys. They
    are skipped here until multi-labeler consensus is decided, so every
    result built on the original labels stays reproducible.
    """
    return {k: v for k, v in votes.items() if isinstance(v, str)}


def load_votes_and_chips(votes: dict, chips: list[dict]) -> list[dict]:
    """Joins vote labels onto chip metadata, dropping 'unsure' votes."""
    chip_by_id = {c["chip_id"]: c for c in chips}
    joined = []
    for chip_id, label in single_labeler_votes(votes).items():
        if label == "unsure":
            continue
        chip = chip_by_id[chip_id]
        joined.append({**chip, "human_label": MINING if label == "mining" else NOT_MINING})
    return joined


def stratified_split(labeled_chips: list[dict], train_fraction: float = 0.5, seed: int = 0):
    """Splits labeled_chips into (train, test), stratified by category so
    both halves see every category and every label."""
    rng = np.random.default_rng(seed)
    by_category = {}
    for c in labeled_chips:
        by_category.setdefault(c["category"], []).append(c)

    train, test = [], []
    for cat, items in by_category.items():
        idx = rng.permutation(len(items))
        n_train = max(1, round(len(items) * train_fraction))
        train.extend(items[i] for i in idx[:n_train])
        test.extend(items[i] for i in idx[n_train:])
    return train, test


def apply_hard_case_labels(labels: np.ndarray, chips: list[dict], window: int = 4) -> np.ndarray:
    """Overlays human labels onto a (H,W) label array, overriding whatever
    concession-based label was there. `window` pixels on each side of the
    chip center get the human label (a small neighborhood, not the full
    64px chip, since the wider chip may not be homogeneous)."""
    labels = labels.copy()
    h, w = labels.shape
    for c in chips:
        r, col = c["row"], c["col"]
        r0, r1 = max(r - window, 0), min(r + window + 1, h)
        c0, c1 = max(col - window, 0), min(col + window + 1, w)
        labels[r0:r1, c0:c1] = c["human_label"]
    return labels


def evaluate_on_chips(proba_map: np.ndarray, chips: list[dict], threshold: float = 0.5) -> dict:
    """Confusion matrix + metrics comparing model predictions at each
    chip's center pixel against its human label."""
    tp = fp = tn = fn = 0
    for c in chips:
        pred = MINING if proba_map[c["row"], c["col"]] >= threshold else NOT_MINING
        actual = c["human_label"]
        if pred == MINING and actual == MINING: tp += 1
        elif pred == MINING and actual == NOT_MINING: fp += 1
        elif pred == NOT_MINING and actual == NOT_MINING: tn += 1
        else: fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }
