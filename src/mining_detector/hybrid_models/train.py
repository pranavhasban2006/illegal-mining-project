"""Trains and evaluates all 5 hybrid architectures on the real (n=32)
hard-case chip dataset. Proof-of-concept scale by necessity -- see
dataset.py and models.py docstrings for what that does and doesn't mean.
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import dataset as ds
from .models import MODEL_REGISTRY, SIAMESE_MODELS

torch.manual_seed(0)


def _evaluate(model, loader, is_siamese: bool):
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for batch in loader:
            if is_siamese:
                x0, x1, y = batch
                logits = model(x0, x1)
            else:
                x, y = batch
                logits = model(x)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            for p, t in zip(preds.tolist(), y.tolist()):
                if p == 1 and t == 1: tp += 1
                elif p == 1 and t == 0: fp += 1
                elif p == 0 and t == 0: tn += 1
                else: fn += 1
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train_one(name: str, epochs: int = 30, lr: float = 1e-3):
    records = ds.load_labeled_chips()
    train_records, test_records = ds.stratified_train_test(records, test_fraction=0.25, seed=0)

    is_siamese = name in SIAMESE_MODELS
    if is_siamese:
        train_ds, test_ds = ds.PairedChipDataset(train_records), ds.PairedChipDataset(test_records)
    else:
        train_ds, test_ds = ds.ChipDataset(train_records), ds.ChipDataset(test_records)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = MODEL_REGISTRY[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if is_siamese:
                x0, x1, y = batch
                logits = model(x0, x1)
            else:
                x, y = batch
                logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    train_metrics = _evaluate(model, train_loader, is_siamese)
    test_metrics = _evaluate(model, test_loader, is_siamese)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "model": name,
        "n_params": n_params,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "train": train_metrics,
        "test": test_metrics,
    }


def train_all(epochs: int = 30):
    results = {}
    for name in MODEL_REGISTRY:
        print(f"Training {name}...")
        result = train_one(name, epochs=epochs)
        results[name] = result
        t = result["test"]
        print(f"  params={result['n_params']:,}  test: acc={t['accuracy']:.3f} "
              f"f1={t['f1']:.3f} precision={t['precision']:.3f} recall={t['recall']:.3f} "
              f"(n={t['n']}: TP={t['tp']} FP={t['fp']} TN={t['tn']} FN={t['fn']})")
    return results


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hybrid_models_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hybrid_results.json")

    results = train_all()
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
