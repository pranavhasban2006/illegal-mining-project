"""Real (if tiny) labeled image dataset for the 5 hybrid architectures.

Source: the 40 hand-labeled hard-case chips from data/hard_cases/ (see
../hard_case_labels.py and the labeling tool). This is a genuine, human-
labeled dataset -- not synthetic -- but it is small (32 usable labels after
excluding "unsure") and has no pixel-level masks or bounding boxes, only a
single mining/not-mining label per 64x64 chip. That shapes what these
architectures can honestly be evaluated on: image classification, not the
segmentation/detection tasks some of them (YOLO, U-Net) were originally
designed for. Each model in models.py is adapted with a classification
head; segmentation-style decoders are kept for architectural fidelity but
are not supervised, since no mask labels exist to train them against.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

HARD_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hard_cases")
T0_CHIPS_DIR = os.path.join(HARD_CASES_DIR, "t0_chips")

LABEL_MAP = {"mining": 1, "not_mining": 0}


def load_labeled_chips() -> list[dict]:
    """Returns [{chip_id, category, label, t1_path, t0_path}], unsure excluded."""
    chips = json.load(open(os.path.join(HARD_CASES_DIR, "chips_data.json")))
    votes = json.load(open(os.path.join(HARD_CASES_DIR, "votes.json")))
    chip_by_id = {c["chip_id"]: c for c in chips}

    records = []
    for chip_id, label in votes.items():
        if label not in LABEL_MAP:
            continue
        chip = chip_by_id[chip_id]
        records.append({
            "chip_id": chip_id,
            "category": chip["category"],
            "label": LABEL_MAP[label],
            "t1_path": os.path.join(HARD_CASES_DIR, f"{chip_id}.png"),
            "t0_path": os.path.join(T0_CHIPS_DIR, f"{chip_id}.png"),
        })
    return records


def stratified_train_test(records: list[dict], test_fraction: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    by_label = {0: [], 1: []}
    for r in records:
        by_label[r["label"]].append(r)

    train, test = [], []
    for label, items in by_label.items():
        idx = rng.permutation(len(items))
        n_test = max(1, round(len(items) * test_fraction))
        test.extend(items[i] for i in idx[:n_test])
        train.extend(items[i] for i in idx[n_test:])
    return train, test


def _load_image_tensor(path: str) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)


class ChipDataset(Dataset):
    """Single-date (t1) chips -- used by the single-image architectures."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = _load_image_tensor(r["t1_path"])
        y = torch.tensor(r["label"], dtype=torch.float32)
        return x, y


class PairedChipDataset(Dataset):
    """Bi-temporal (t0, t1) pairs -- used by the Siamese architecture."""

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x0 = _load_image_tensor(r["t0_path"])
        x1 = _load_image_tensor(r["t1_path"])
        y = torch.tensor(r["label"], dtype=torch.float32)
        return x0, x1, y
