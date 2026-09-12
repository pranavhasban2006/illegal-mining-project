"""Retrains the architectures that can use it against the real Open-Pit
Mining Areas Dataset (openpit_dataset.py) instead of the 32-chip toy set:

- unet_transformer, resnet_attention_unet: real per-pixel segmentation,
  evaluated with Dice/IoU against real masks on held-out tiles.
- yolo_vit_lstm, cnn_bilstm: classification on patch-level labels derived
  from the same real masks -- far larger n than the 32-chip set.

siamese_cnn_transformer is NOT retrained here: it needs genuine bi-temporal
pairs, and the Open-Pit dataset is single-date. It stays on the 32 real
hard-case chip pairs (train.py) -- the right data for what it actually is,
not a forced fit to data that doesn't match its design.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import openpit_dataset as opd
from .models import MODEL_REGISTRY

torch.manual_seed(0)

SEG_MODELS = ["unet_transformer", "resnet_attention_unet"]
CLS_MODELS = ["yolo_vit_lstm", "cnn_bilstm"]


def dice_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    dice = (2 * intersection + eps) / (pred.sum() + target.sum() + eps)
    iou = (intersection + eps) / (union + eps)
    return dice.item(), iou.item()


def train_segmentation(name: str, train_loader, test_loader, epochs: int = 15, lr: float = 1e-3):
    model = MODEL_REGISTRY[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x, return_map=True)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x, return_map=True)
            d, i = dice_iou(logits, y)
            dices.append(d)
            ious.append(i)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": name, "task": "segmentation", "n_params": n_params,
        "test_dice_mean": sum(dices) / len(dices), "test_iou_mean": sum(ious) / len(ious),
        "n_test_patches": len(test_loader.dataset),
    }


def _evaluate_cls(model, loader):
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for x, y in loader:
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


def train_classification(name: str, train_loader, test_loader, epochs: int = 15, lr: float = 1e-3):
    model = MODEL_REGISTRY[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    train_metrics = _evaluate_cls(model, train_loader)
    test_metrics = _evaluate_cls(model, test_loader)
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": name, "task": "classification", "n_params": n_params,
        "n_train": len(train_loader.dataset), "n_test": len(test_loader.dataset),
        "train": train_metrics, "test": test_metrics,
    }


def run():
    print("Building patch datasets from the Open-Pit Mining Areas Dataset...")
    train_patches, test_patches = opd.build_patch_lists()
    print(f"  {len(train_patches)} train patches, {len(test_patches)} test patches "
          f"(held out entirely: {sorted(opd.TEST_TILES)})")

    results = {}

    seg_train = DataLoader(opd.SegPatchDataset(train_patches), batch_size=16, shuffle=True)
    seg_test = DataLoader(opd.SegPatchDataset(test_patches), batch_size=16, shuffle=False)
    for name in SEG_MODELS:
        print(f"Training {name} (real segmentation)...")
        result = train_segmentation(name, seg_train, seg_test)
        results[name] = result
        print(f"  params={result['n_params']:,} test_dice={result['test_dice_mean']:.3f} "
              f"test_iou={result['test_iou_mean']:.3f} (n={result['n_test_patches']} patches)")

    cls_train = DataLoader(opd.ClsPatchDataset(train_patches), batch_size=16, shuffle=True)
    cls_test = DataLoader(opd.ClsPatchDataset(test_patches), batch_size=16, shuffle=False)
    for name in CLS_MODELS:
        print(f"Training {name} (classification on real patch labels)...")
        result = train_classification(name, cls_train, cls_test)
        results[name] = result
        t = result["test"]
        print(f"  params={result['n_params']:,} test: acc={t['accuracy']:.3f} f1={t['f1']:.3f} "
              f"precision={t['precision']:.3f} recall={t['recall']:.3f} (n={t['n']})")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hybrid_models_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "openpit_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


if __name__ == "__main__":
    run()
