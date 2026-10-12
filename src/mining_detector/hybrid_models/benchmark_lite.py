"""Compare each full model with its lite version: size, CPU speed, and test score.

Usage:
    python -m mining_detector.hybrid_models.benchmark_lite

For each model it reports:
    params         number of trainable weights
    latency_ms     median time for one forward pass on a batch of 8 patches, on this CPU
    score          the test score from the results file, if that model was trained
                   (IoU for segmentation, F1 for classification). "not trained" otherwise.

The latency is measured on this machine only, so compare the full and lite columns
against each other rather than against numbers from another computer.
Output: data/hybrid_models_results/benchmark_lite.json
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from mining_detector.hybrid_models.models import (
    ALL_MODELS, SEGMENTATION_MODELS, SIAMESE_MODELS, build_model, count_parameters,
)

REPO_DIR = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_DIR / "data" / "hybrid_models_results"
BATCH = 8
REPEATS = 10


def latency_ms(model: torch.nn.Module, in_ch: int, size: int, siamese: bool) -> float:
    """Median milliseconds for one forward pass, after a warm-up run."""
    model.eval()
    x = torch.randn(BATCH, in_ch, size, size)
    args = (x, x) if siamese else (x,)
    times = []
    with torch.no_grad():
        model(*args)  # warm-up
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            model(*args)
            times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 2)


def load_scores() -> dict:
    """Map (name, lite) to the score from openpit_results.json or standin_results.json, if any."""
    scores = {}
    for fname, source in (("openpit_results.json", "open_pit"), ("standin_results.json", "standin")):
        path = RESULTS_DIR / fname
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("results", []):
            key = (row["name"], row["lite"])
            if key not in scores or source == "open_pit":
                metric = row.get("iou", row.get("f1"))
                scores[key] = {"metric": "IoU" if "iou" in row else "F1", "value": metric,
                               "source": source}
    return scores


def benchmark(in_ch: int = 6, size: int = 64) -> list[dict]:
    """Build every model in full and lite form and measure it. Returns one row per model and size."""
    scores = load_scores()
    rows = []
    for name in ALL_MODELS:
        for lite in (False, True):
            model = build_model(name, in_ch=in_ch, lite=lite, image_size=size)
            siamese = name in SIAMESE_MODELS
            row = {
                "name": name,
                "variant": "lite" if lite else "full",
                "params": count_parameters(model),
                "latency_ms": latency_ms(model, in_ch, size, siamese),
                "task": "segmentation" if name in SEGMENTATION_MODELS
                        else ("bi-temporal change" if siamese else "patch classification"),
            }
            s = scores.get((name, lite))
            row["score"] = s["value"] if s else None
            row["score_metric"] = s["metric"] if s else None
            row["score_source"] = s["source"] if s else "not trained"
            rows.append(row)
            print(f"{name:24s} {row['variant']:5s} params={row['params']:6d} "
                  f"latency={row['latency_ms']:7.2f} ms  score={row['score']} ({row['score_source']})")
    return rows


def main() -> int:
    rows = benchmark()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "benchmark_lite.json"
    out.write_text(json.dumps({"batch": BATCH, "repeats": REPEATS,
                               "cpu": "measured on the machine that ran this script",
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
