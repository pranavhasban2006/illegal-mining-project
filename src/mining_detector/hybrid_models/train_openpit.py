"""Train the hybrid models and save honest test scores.

Usage (from the repo root, after the Open-Pit folder is in place):

    python -m mining_detector.hybrid_models.train_openpit
    python -m mining_detector.hybrid_models.train_openpit --epochs 10 --models CNNBiLSTM

If the Open-Pit data is missing the script stops with a clear message. Pass --standin to
train on synthetic stand-in data instead. Those results are written to standin_results.json,
are flagged demo_only, and must not be quoted as real accuracy.

What is measured
----------------
Segmentation (UNetTransformer, ResNetAttentionUNet): intersection-over-union (IoU) and
Dice on the test tiles, per pixel.
Classification (YoloViTLSTM, CNNBiLSTM): accuracy and F1 on test patches, with "mining" as
the positive class.
Siamese (SiameseCNNTransformer): accuracy and F1 on test patches from held-out synthetic
scenes. Always demo_only, because the synthetic data is not a real-world benchmark.

Class imbalance is handled with weighted losses. Mining pixels and patches are rare, so an
unweighted loss would let a model that always predicts "no mining" look good.
Training stops early when validation loss has not improved for PATIENCE epochs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mining_detector.hybrid_models.models import (
    ALL_MODELS, CLASSIFICATION_MODELS, SEGMENTATION_MODELS, SIAMESE_MODELS,
    build_model, count_parameters,
)
from mining_detector.hybrid_models.openpit_dataset import (
    DEFAULT_DATA_DIR, DatasetMissing, PatchDataset, Splits, build_splits, load_open_pit,
    patch_labels, standin_tiles, tile_pair,
)

REPO_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_DIR / "data" / "hybrid_models_results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
DEFAULT_EPOCHS = 15
PATIENCE = 4
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


# ------------------------------------------------------------------ metrics -----

def segmentation_scores(logits: np.ndarray, target: np.ndarray) -> dict:
    """IoU and Dice over all pixels. A score is 0.0 if the denominator is 0 (nothing predicted or present)."""
    pred = logits > 0.0  # logit 0 is probability 0.5
    tgt = target.astype(bool)
    tp = float(np.logical_and(pred, tgt).sum())
    fp = float(np.logical_and(pred, ~tgt).sum())
    fn = float(np.logical_and(~pred, tgt).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"iou": round(iou, 4), "dice": round(dice, 4)}


def classification_scores(pred: np.ndarray, target: np.ndarray) -> dict:
    """Accuracy and F1 for the positive class (1 = mining)."""
    acc = float((pred == target).mean()) if len(target) else 0.0
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"accuracy": round(acc, 4), "f1": round(f1, 4)}


def _batches(x: np.ndarray, y: np.ndarray, shuffle: bool, seed: int = 0):
    """Yield (x, y) mini-batches as tensors."""
    idx = np.arange(len(x))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx), BATCH_SIZE):
        sel = idx[start:start + BATCH_SIZE]
        yield torch.from_numpy(x[sel]), torch.from_numpy(y[sel])


# ----------------------------------------------------------------- training -----

def _fit(model: nn.Module, forward_loss, train_iter_fn, val_loss_fn, epochs: int) -> dict:
    """Generic loop: train, check validation loss each epoch, keep the best weights, stop early."""
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = [forward_loss(opt, batch) for batch in train_iter_fn(epoch)]
        model.eval()
        with torch.no_grad():
            val_loss = val_loss_fn()
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                        "val_loss": round(val_loss, 5)})
        if val_loss < best["val_loss"] - 1e-4:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "epochs_run": len(history), "history": history}


def _pos_weight(y: np.ndarray) -> float:
    """Weight that makes the rare positive pixels count as much as the negative ones."""
    pos = float(y.sum())
    neg = float(y.size - pos)
    return neg / pos if pos > 0 else 1.0


def train_segmentation(name: str, lite: bool, data: Splits, epochs: int) -> tuple[nn.Module, dict]:
    """Train a per-pixel model with a positive-weighted BCE loss. Returns (model, scores)."""
    in_ch = data.train_x.shape[1]
    model = build_model(name, in_ch=in_ch, lite=lite, image_size=data.train_x.shape[-1])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(_pos_weight(data.train_y)))

    def forward_loss(opt, batch):
        x, y = batch
        logits = model(x)
        loss = loss_fn(logits, y.unsqueeze(1).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def val_loss():
        losses = [loss_fn(model(x), y.unsqueeze(1).float()).item()
                  for x, y in _batches(data.val_x, data.val_y, shuffle=False)]
        return float(np.mean(losses)) if losses else float("nan")

    def train_iter(epoch):
        ds = PatchDataset(data.train_x, data.train_y, augment_on=True, seed=epoch)
        xs = np.stack([ds[i][0] for i in range(len(ds))])
        ys = np.stack([ds[i][1] for i in range(len(ds))])
        return _batches(xs, ys, shuffle=True, seed=epoch)

    info = _fit(model, forward_loss, train_iter, val_loss, epochs)
    model.eval()
    with torch.no_grad():
        logits = np.concatenate([model(x).numpy()[:, 0] for x, _ in
                                 _batches(data.test_x, data.test_y, shuffle=False)])
    return model, {**info, **segmentation_scores(logits, data.test_y)}


def train_classification(name: str, lite: bool, data: Splits, epochs: int) -> tuple[nn.Module, dict]:
    """Train a patch classifier with class-weighted cross-entropy. Returns (model, scores)."""
    in_ch = data.train_x.shape[1]
    y_tr = patch_labels(data.train_y)
    y_va = patch_labels(data.val_y)
    y_te = patch_labels(data.test_y)
    counts = np.bincount(y_tr, minlength=2).astype(float)
    weights = counts.sum() / (2 * np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    model = build_model(name, in_ch=in_ch, lite=lite, image_size=data.train_x.shape[-1])

    def forward_loss(opt, batch):
        x, y = batch
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def val_loss():
        losses = [loss_fn(model(x), y).item() for x, y in _batches(data.val_x, y_va, shuffle=False)]
        return float(np.mean(losses)) if losses else float("nan")

    def train_iter(epoch):
        return _batches(data.train_x, y_tr, shuffle=True, seed=epoch)

    info = _fit(model, forward_loss, train_iter, val_loss, epochs)
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(x).argmax(1).numpy() for x, _ in
                               _batches(data.test_x, y_te, shuffle=False)])
    return model, {**info, **classification_scores(pred, y_te)}


def siamese_scenes(n_scenes: int, seed: int = 0):
    """Synthetic bi-temporal patches, split by whole scene: last scene for test, one for validation."""
    from mining_detector.synthetic import make_synthetic_scene

    bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
    scenes = []
    for k in range(n_scenes):
        s = make_synthetic_scene(seed=seed + k, size=192)
        t0 = np.stack([s.bands_t0[b][:192, :192] for b in bands]).astype(np.float32)
        t1 = np.stack([s.bands_t1[b][:192, :192] for b in bands]).astype(np.float32)
        scenes.append((t0, t1, s.truth_new_mask[:192, :192].astype(np.uint8)))
    test_scene, val_scene = scenes[-1], scenes[-2]
    train_scenes = scenes[:-2]

    def cut(items):
        t0s, t1s, ys = [], [], []
        for t0, t1, m in items:
            p0, pm = tile_pair(t0, m)
            p1, _ = tile_pair(t1, m)
            t0s.append(p0)
            t1s.append(p1)
            ys.append(patch_labels(pm))
        return np.concatenate(t0s), np.concatenate(t1s), np.concatenate(ys)

    tr = cut(train_scenes)
    va = cut([val_scene])
    te = cut([test_scene])
    return tr, va, te


def train_siamese(name: str, lite: bool, epochs: int, n_scenes: int = 8) -> tuple[nn.Module, dict]:
    """Train the two-date change classifier on synthetic scenes. Demo only."""
    (tr0, tr1, tr_y), (va0, va1, va_y), (te0, te1, te_y) = siamese_scenes(n_scenes)
    counts = np.bincount(tr_y, minlength=2).astype(float)
    weights = counts.sum() / (2 * np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    model = build_model(name, in_ch=tr0.shape[1], lite=lite, image_size=tr0.shape[-1])

    def forward_loss(opt, batch):
        x0, x1, y = batch
        loss = loss_fn(model(x0, x1), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    def val_loss():
        losses = [loss_fn(model(torch.from_numpy(a), torch.from_numpy(b)), torch.from_numpy(c)).item()
                  for a, b, c in [(va0, va1, va_y)]]
        return float(np.mean(losses))

    def train_iter(epoch):
        rng = np.random.default_rng(epoch)
        idx = rng.permutation(len(tr_y))
        for s in range(0, len(idx), BATCH_SIZE):
            sel = idx[s:s + BATCH_SIZE]
            yield (torch.from_numpy(tr0[sel]), torch.from_numpy(tr1[sel]), torch.from_numpy(tr_y[sel]))

    info = _fit(model, forward_loss, train_iter, val_loss, epochs)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(te0), torch.from_numpy(te1)).argmax(1).numpy()
    return model, {**info, **classification_scores(pred, te_y)}


# ----------------------------------------------------------------- driver -------

def _save(model: nn.Module, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), **meta}, path)


def run(models: list[str], epochs: int, standin: bool, lite_too: bool) -> Path:
    """Train every requested model (and its lite version) and write the results file."""
    torch.manual_seed(0)
    if standin:
        tiles = standin_tiles(n_scenes=10)
        source, tag = "synthetic_standin", "standin"
        data = build_splits(tiles, seed=0, source=source)
    else:
        try:
            tiles = load_open_pit(DEFAULT_DATA_DIR)
        except DatasetMissing as err:
            raise SystemExit(f"Cannot train: {err}\nTo test the code without the data, add --standin.")
        source, tag = "open_pit", "openpit"
        data = build_splits(tiles, seed=0, source=source)

    results = []
    for name in models:
        for lite in ([False, True] if lite_too else [False]):
            label = f"{name}{' (lite)' if lite else ''}"
            print(f"Training {label} ...", flush=True)
            t_start = time.time()
            if name in SIAMESE_MODELS:
                model, scores = train_siamese(name, lite, epochs)
                task, demo_only = "bi-temporal change", True
            elif name in SEGMENTATION_MODELS:
                model, scores = train_segmentation(name, lite, data, epochs)
                task, demo_only = "segmentation", standin
            else:
                model, scores = train_classification(name, lite, data, epochs)
                task, demo_only = "patch classification", standin
            seconds = round(time.time() - t_start, 1)
            row = {"name": name, "lite": lite, "task": task, "params": count_parameters(model),
                   "train_seconds": seconds, **scores, "demo_only": demo_only}
            results.append(row)
            print(f"  done in {seconds}s, best epoch {scores['best_epoch']}: "
                  + ", ".join(f"{k}={v}" for k, v in scores.items() if k in ("iou", "dice", "accuracy", "f1")))
            weight_name = f"{tag}_{name}{'_lite' if lite else ''}.pt"
            _save(model, {"name": name, "lite": lite, "in_ch": data.train_x.shape[1],
                          "image_size": data.train_x.shape[-1], "mean": data.mean.tolist(),
                          "std": data.std.tolist(), "task": task, "source": source},
                  WEIGHTS_DIR / weight_name)

    out = {
        "data_source": source,
        "demo_only": standin,
        "epochs_cap": epochs,
        "patience": PATIENCE,
        "patch_size": int(data.train_x.shape[-1]),
        "n_bands": int(data.train_x.shape[1]),
        "test_tile_ids": data.test_tile_ids,
        "n_train_patches": int(len(data.train_x)),
        "n_test_patches": int(len(data.test_x)),
        "honesty_note": (
            "Test patches come from only the held-out source tiles. This measures how well the "
            "models fit new patches from the same scenes, not how well they generalise to new places."
        ),
        "results": results,
    }
    path = RESULTS_DIR / ("standin_results.json" if standin else "openpit_results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train hybrid models on the Open-Pit dataset.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="maximum epochs per model")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    parser.add_argument("--standin", action="store_true",
                        help="train on synthetic stand-in data (demo only, not real results)")
    parser.add_argument("--no-lite", action="store_true", help="skip the lite variants")
    args = parser.parse_args(argv)
    run(args.models, args.epochs, args.standin, lite_too=not args.no_lite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
