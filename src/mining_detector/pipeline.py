"""End-to-end pipeline: scenes in, alert files and a report out.

Public entry points (the contract with the app and the CLI):
    run_demo(out_dir="outputs") -> RunResult
        Runs on a built-in synthetic scene with a known answer. No internet needed.
    run_live(...) -> RunResult
        Runs on real Sentinel-2 data (added in a later phase).

Both write, into out_dir:
    alerts.geojson, report.json, change_map.png, probability_t1.png
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # no GUI needed; the PNGs are written straight to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from affine import Affine  # noqa: E402

from .alerts import build_feature_collection, mask_to_polygons, write_geojson  # noqa: E402
from .change_detection import (  # noqa: E402
    min_area_filter,
    morphological_cleanup,
    new_disturbance_mask,
    rule_mining_probability,
)
from .config import Config  # noqa: E402
from .indices import compute_feature_stack  # noqa: E402
from .licensing import classify_alerts  # noqa: E402

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class RunResult:
    """What a run produced. The app reads these fields."""

    alerts_path: Path
    report_path: Path
    summary: dict
    metrics: dict = field(default_factory=dict)


def _noop_progress(fraction: float, message: str) -> None:
    """Default progress callback: log the step and do nothing else."""
    log.info("[%3d%%] %s", int(fraction * 100), message)


def _report(progress: ProgressFn | None) -> ProgressFn:
    return progress if progress is not None else _noop_progress


def _save_quicklooks(
    out_dir: Path,
    stack0: np.ndarray,
    stack1: np.ndarray,
    prob1: np.ndarray,
    new_mask: np.ndarray,
) -> tuple[Path, Path]:
    """Write change_map.png (true-colour t1 with new disturbance outlined) and probability_t1.png."""
    # Bands in the stack are B02, B03, B04 at indices 0, 1, 2; true colour is R=B04, G=B03, B=B02.
    def rgb(stack: np.ndarray) -> np.ndarray:
        img = stack[..., [2, 1, 0]]
        lo, hi = np.percentile(img, 2), np.percentile(img, 98)
        return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb(stack0))
    ax[0].set_title("Before (t0), true colour")
    ax[1].imshow(rgb(stack1))
    ax[1].contour(new_mask.astype(float), levels=[0.5], colors="red", linewidths=1)
    ax[1].set_title("After (t1), red = new disturbance")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    change_path = out_dir / "change_map.png"
    fig.savefig(change_path, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(prob1, cmap="magma", vmin=0, vmax=1)
    ax.set_title("Mining probability at t1")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    prob_path = out_dir / "probability_t1.png"
    fig.savefig(prob_path, dpi=120)
    plt.close(fig)
    return change_path, prob_path


def detect_and_write(
    *,
    stack0: np.ndarray,
    stack1: np.ndarray,
    prob0: np.ndarray,
    prob1: np.ndarray,
    transform: Affine,
    crs: str,
    concessions_utm: gpd.GeoDataFrame | None,
    cfg: Config,
    bbox: tuple[float, float, float, float],
    data_sources: list[str],
    caveats: list[str],
    model_info: dict,
    progress: ProgressFn | None = None,
) -> RunResult:
    """Shared core of the demo and live runs.

    Takes two aligned feature stacks and their mining-probability maps, finds new
    disturbance, labels it, and writes the four output files.
    """
    report_cb = _report(progress)
    t_start = time.time()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_cb(0.6, "Finding new disturbance (mining at t1, not at t0)")
    raw_mask = new_disturbance_mask(prob0, prob1, cfg.p_high, cfg.p_low)
    clean = morphological_cleanup(raw_mask)
    pixel_area_m2 = abs(transform.a * transform.e)
    keep = min_area_filter(clean, pixel_area_m2, cfg.min_area_ha)

    report_cb(0.75, "Vectorizing alerts and checking concession licences")
    polygons = mask_to_polygons(keep, transform, crs, prob=prob1)
    if concessions_utm is not None and not concessions_utm.empty:
        concessions_utm = concessions_utm.to_crs(crs)
    labels, notes = classify_alerts(polygons, concessions_utm, buffer_m=cfg.buffer_m)
    caveats = list(caveats) + notes

    report_cb(0.85, "Writing alerts.geojson, report.json and quicklook images")
    fc = build_feature_collection(polygons, labels, detected_on=cfg.t1)
    alerts_path = write_geojson(fc, out_dir / "alerts.geojson")

    by_status = {"legal": 0.0, "disputed": 0.0, "illegal": 0.0}
    for poly_area, status in zip(polygons["area_ha"], labels):
        by_status[status] += float(poly_area)
    total_ha = float(sum(by_status.values()))

    _save_quicklooks(out_dir, stack0, stack1, prob1, keep)

    summary = {
        "n_alerts": len(polygons),
        "total_new_disturbance_ha": round(total_ha, 2),
        "by_legal_status": {k: round(v, 2) for k, v in by_status.items()},
        "runtime_s": round(time.time() - t_start, 2),
    }
    report = {
        "aoi_bbox": list(bbox),
        "t0": cfg.t0,
        "t1": cfg.t1,
        "n_alerts": summary["n_alerts"],
        "total_new_disturbance_ha": summary["total_new_disturbance_ha"],
        "by_legal_status": summary["by_legal_status"],
        "model": model_info,
        "data_sources": data_sources,
        "caveats": caveats,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report_cb(1.0, f"Done: {summary['n_alerts']} new disturbance area(s) found")
    return RunResult(alerts_path=alerts_path, report_path=report_path, summary=summary)


def run_demo(out_dir: str | Path = "outputs", progress: ProgressFn | None = None) -> RunResult:
    """Run the full pipeline on the built-in synthetic scene.

    The synthetic scene has one licensed mine (should NOT be alerted) and two new
    unlicensed sites (should be alerted as 'illegal'). Metrics compare the output
    against that known answer.

    Args:
        out_dir: folder for the output files.
        progress: optional callback progress(fraction, message).

    Returns:
        RunResult with paths, a summary, and pixel-level metrics against the truth.
    """
    from .synthetic import make_synthetic_scene  # local import keeps the core lighter

    report_cb = _report(progress)
    report_cb(0.05, "Building the synthetic two-date scene (no internet needed)")
    scene = make_synthetic_scene(seed=42)
    cfg = Config(out_dir=Path(out_dir), t0=scene.t0, t1=scene.t1, min_area_ha=0.5)
    cfg.validate()

    report_cb(0.2, "Computing spectral indices for both dates")
    stack0, names = compute_feature_stack(scene.bands_t0)
    stack1, _ = compute_feature_stack(scene.bands_t1)

    report_cb(0.4, "Scoring mining likelihood with the rule-based demo model")
    prob0 = rule_mining_probability(stack0)
    prob1 = rule_mining_probability(stack1)

    model_info = {
        "best": "rule-based spectral score (demo; no trained model)",
        "candidates": [],
        "feature_importance": {},
    }
    caveats = [
        "DEMO ONLY: the data are synthetic, generated by the pipeline to test it. Results do not describe any real place.",
        "The demo model is a fixed rule, not trained on labelled data, so no accuracy claim applies.",
    ]
    result = detect_and_write(
        stack0=stack0, stack1=stack1, prob0=prob0, prob1=prob1,
        transform=scene.transform, crs=scene.crs,
        concessions_utm=scene.concessions, cfg=cfg,
        bbox=_bbox_lonlat(scene.transform, scene.shape, scene.crs),
        data_sources=["synthetic scene generated by mining_detector.synthetic"],
        caveats=caveats, model_info=model_info, progress=progress,
    )

    # Compare with the known answer. Pixel-level: the predicted new-disturbance mask vs truth.
    predicted = _predicted_mask(prob0, prob1, cfg, scene.transform)
    truth = scene.truth_new_mask
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result.metrics = {
        "truth": "synthetic ground truth (known by construction)",
        "pixel_precision": round(precision, 4),
        "pixel_recall": round(recall, 4),
        "pixel_f1": round(f1, 4),
        "expected_alerts": 2,
        "found_alerts": result.summary["n_alerts"],
    }
    return result


def _bbox_lonlat(
    transform: Affine, shape: tuple[int, int], crs: str
) -> tuple[float, float, float, float]:
    """Return the raster extent as (west, south, east, north) in EPSG:4326 degrees."""
    from pyproj import Transformer

    h, w = shape
    x0, y0 = transform.c, transform.f                              # top-left corner
    x1, y1 = transform.c + w * transform.a, transform.f + h * transform.e  # bottom-right
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform([min(x0, x1), max(x0, x1)], [min(y0, y1), max(y0, y1)])
    return (round(lon[0], 6), round(lat[0], 6), round(lon[1], 6), round(lat[1], 6))


def _predicted_mask(prob0, prob1, cfg: Config, transform: Affine) -> np.ndarray:
    """Recompute the cleaned change mask (same steps as detect_and_write) for scoring."""
    raw = new_disturbance_mask(prob0, prob1, cfg.p_high, cfg.p_low)
    clean = morphological_cleanup(raw)
    return min_area_filter(clean, abs(transform.a * transform.e), cfg.min_area_ha)


def _date_ok(value: str) -> str:
    """Validate a YYYY-MM-DD string and return it."""
    date.fromisoformat(value)
    return value
