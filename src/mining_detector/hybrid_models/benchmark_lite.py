"""Benchmarks full-size vs. lightweight variants of all 5 architectures:
parameter count, CPU inference latency, and accuracy/IoU on the same real
data each architecture already trains on (openpit_dataset.py for the 4
single-date architectures, dataset.py's 32-chip set for the Siamese model).

Addresses MineFormer's and TransSiamUNet's future-work item (both cited in
Project Review 1): "further reduction of computational cost... for
real-time or edge-device deployment."
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import dataset as ds
from . import openpit_dataset as opd
from .models import build_model, SIAMESE_MODELS
from .train_openpit import SEG_MODELS, CLS_MODELS, dice_iou
from .train import _evaluate as _evaluate_cls_chip

torch.manual_seed(0)

N_LATENCY_RUNS = 30


def _measure_latency(model: nn.Module, sample_inputs, batch_size: int = 1) -> float:
    """Mean single-batch CPU inference latency in milliseconds."""
    model.eval()
    inputs = [t[:batch_size] for t in sample_inputs]
    with torch.no_grad():
        for _ in range(3):  # warm-up
            model(*inputs)
        start = time.perf_counter()
        for _ in range(N_LATENCY_RUNS):
            model(*inputs)
        elapsed = time.perf_counter() - start
    return (elapsed / N_LATENCY_RUNS) * 1000


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _train_and_eval_seg(name: str, lite: bool, train_loader, test_loader, epochs=15, lr=1e-3):
    model = build_model(name, lite=lite)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x, return_map=True), y)
            loss.backward()
            optimizer.step()

    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for x, y in test_loader:
            d, i = dice_iou(model(x, return_map=True), y)
            dices.append(d)
            ious.append(i)
    return model, {"test_dice_mean": sum(dices) / len(dices), "test_iou_mean": sum(ious) / len(ious)}


def _train_and_eval_cls(name: str, lite: bool, train_loader, test_loader, epochs=15, lr=1e-3):
    model = build_model(name, lite=lite)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    from .train_openpit import _evaluate_cls
    metrics = _evaluate_cls(model, test_loader)
    return model, metrics


def _train_and_eval_siamese(lite: bool, train_records, test_records, epochs=30, lr=1e-3):
    train_loader = DataLoader(ds.PairedChipDataset(train_records), batch_size=8, shuffle=True)
    test_loader = DataLoader(ds.PairedChipDataset(test_records), batch_size=8, shuffle=False)

    model = build_model("siamese_cnn_transformer", lite=lite)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for x0, x1, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x0, x1), y)
            loss.backward()
            optimizer.step()

    metrics = _evaluate_cls_chip(model, test_loader, is_siamese=True)
    sample = next(iter(test_loader))
    latency = _measure_latency(model, (sample[0], sample[1]), batch_size=1)
    return model, metrics, latency


def run():
    print("Loading data...")
    train_patches, test_patches = opd.build_patch_lists()
    seg_train = DataLoader(opd.SegPatchDataset(train_patches), batch_size=16, shuffle=True)
    seg_test = DataLoader(opd.SegPatchDataset(test_patches), batch_size=16, shuffle=False)
    cls_train = DataLoader(opd.ClsPatchDataset(train_patches), batch_size=16, shuffle=True)
    cls_test = DataLoader(opd.ClsPatchDataset(test_patches), batch_size=16, shuffle=False)

    records = ds.load_labeled_chips()
    train_records, test_records = ds.stratified_train_test(records, test_fraction=0.25, seed=0)

    results = {}
    sample_image_batch = next(iter(cls_test))[0]

    for name in SEG_MODELS:
        print(f"\n{name}:")
        row = {}
        for lite in (False, True):
            tag = "lite" if lite else "full"
            model, metrics = _train_and_eval_seg(name, lite, seg_train, seg_test)
            latency = _measure_latency(model, (sample_image_batch,))
            n_params = _param_count(model)
            row[tag] = {**metrics, "n_params": n_params, "latency_ms": latency}
            print(f"  {tag:5s}: params={n_params:,} latency={latency:.2f}ms "
                  f"iou={metrics['test_iou_mean']:.3f} dice={metrics['test_dice_mean']:.3f}")
        results[name] = row

    for name in CLS_MODELS:
        print(f"\n{name}:")
        row = {}
        for lite in (False, True):
            tag = "lite" if lite else "full"
            model, metrics = _train_and_eval_cls(name, lite, cls_train, cls_test)
            latency = _measure_latency(model, (sample_image_batch,))
            n_params = _param_count(model)
            row[tag] = {**metrics, "n_params": n_params, "latency_ms": latency}
            print(f"  {tag:5s}: params={n_params:,} latency={latency:.2f}ms "
                  f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}")
        results[name] = row

    print(f"\nsiamese_cnn_transformer:")
    row = {}
    for lite in (False, True):
        tag = "lite" if lite else "full"
        model, metrics, latency = _train_and_eval_siamese(lite, train_records, test_records)
        n_params = _param_count(model)
        row[tag] = {**metrics, "n_params": n_params, "latency_ms": latency}
        print(f"  {tag:5s}: params={n_params:,} latency={latency:.2f}ms "
              f"acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f}")
    results["siamese_cnn_transformer"] = row

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hybrid_models_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lite_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return results


if __name__ == "__main__":
    run()
