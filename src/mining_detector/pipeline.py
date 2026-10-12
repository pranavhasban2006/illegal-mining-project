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
    normalize_per_scene,
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


def _stretch_rgb(rgb: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """Make a true-colour image visible: stretch each channel on its own, then apply gamma.

    Reflectance over land is low (0-0.2), so a plain linear view looks almost black.
    """
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        out[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0, 1)
    return out ** gamma


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
        return _stretch_rgb(stack[..., [2, 1, 0]])

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
    validation_points=None,
) -> RunResult:
    """Shared core of the demo and live runs.

    Takes two aligned feature stacks and their mining-probability maps, finds new
    disturbance, labels it, and writes the four output files. If validation_points (a
    GeoDataFrame of hand labels) is given, the alert mask is scored against them and the
    result is stored under "validation" in report.json.
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
    if validation_points is not None and len(validation_points):
        from .validation import evaluate_points

        report["validation"] = evaluate_points(validation_points, keep, transform, crs)
        log.info("Validation against %d check points: F1 %.3f",
                 len(validation_points), report["validation"]["f1"])
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


def default_t0_for(t1: str) -> str:
    """Same calendar date one year before t1 (phenology is comparable on the same date)."""
    d = date.fromisoformat(t1)
    try:
        return d.replace(year=d.year - 1).isoformat()
    except ValueError:  # 29 February
        return d.replace(year=d.year - 1, day=28).isoformat()


def aoi_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Approximate area of the bounding box in square kilometres."""
    w, s, e, n = bbox
    lat_mid = np.radians((s + n) / 2)
    km_lat = (n - s) * 111.32
    km_lon = (e - w) * 111.32 * np.cos(lat_mid)
    return float(km_lat * km_lon)


def run_live(
    aoi_bbox: tuple[float, float, float, float],
    t0_date: str,
    t1_date: str,
    out_dir: str | Path = "outputs",
    concessions_path: str | Path | None = None,
    progress: ProgressFn | None = None,
    max_cloud: float = 20.0,
    max_aoi_km2: float = 400.0,
    hard_cases_path: str | Path | None = None,
    validation_path: str | Path | None = None,
    review_chips: int = 0,
    cache_dir: str | Path = "data/cache",
    external_dir: str | Path = "data/external",
    seed: int = 42,
) -> RunResult:
    """Run the full detector on real Sentinel-2 data.

    Args:
        aoi_bbox: (west, south, east, north) in degrees.
        t0_date: earlier date, "YYYY-MM-DD". Use default_t0_for(t1) for a one-year gap.
        t1_date: later date, "YYYY-MM-DD".
        out_dir: output folder.
        concessions_path: optional concession polygons (GeoJSON or GPKG).
        progress: optional callback progress(fraction, message).
        max_cloud: maximum scene cloud cover in percent.
        max_aoi_km2: refuse larger areas, to keep runtime bounded.
        hard_cases_path: optional CSV/GeoJSON of hand labels (lon, lat, label).
        cache_dir, external_dir: data folders.
        seed: random seed, so repeated runs give identical results.

    Raises:
        ValueError: for a bad bbox, bad dates or an area that is too large.
        RuntimeError: when no usable satellite scene can be found.
        FileNotFoundError: when the mining training labels are missing.
    """
    from .data_sources import search_scenes
    from .detection import MiningDetector
    from .labels import build_labels
    from .model_selection import fit_and_select
    from .preprocess import load_scene, target_grid

    report_cb = _report(progress)
    cfg = Config(
        bbox=tuple(aoi_bbox), t0=t0_date, t1=t1_date, out_dir=Path(out_dir),
        cache_dir=Path(cache_dir), external_dir=Path(external_dir),
        concessions_path=Path(concessions_path) if concessions_path else None,
        max_cloud=max_cloud, max_aoi_km2=max_aoi_km2, seed=seed,
    )
    cfg.validate()
    if t1_date <= t0_date:
        raise ValueError("t1 must be later than t0.")
    area = aoi_area_km2(cfg.bbox)
    if area > max_aoi_km2:
        raise ValueError(
            f"The area is about {area:,.0f} km2, above the limit of {max_aoi_km2:,.0f} km2. "
            "Draw a smaller box, or raise --max-aoi-km2 if you have the time and memory."
        )
    np.random.seed(seed)
    t_start = time.time()
    log.info("Live run: bbox %s, %s vs %s, area %.0f km2", cfg.bbox, t0_date, t1_date, area)

    report_cb(0.02, f"Looking for the least cloudy scene near {t0_date}")
    choice0 = search_scenes(cfg.bbox, t0_date, max_cloud=max_cloud, window_days=cfg.search_window_days)
    report_cb(0.06, f"Looking for the least cloudy scene near {t1_date}")
    choice1 = search_scenes(cfg.bbox, t1_date, max_cloud=max_cloud, window_days=cfg.search_window_days)

    grid = target_grid(cfg.bbox)
    report_cb(0.10, f"Reading t0 bands ({choice0.describe()})")
    scene0 = load_scene(choice0, cfg.bbox, grid=grid, cache_dir=cache_dir)
    report_cb(0.20, f"Reading t1 bands ({choice1.describe()})")
    scene1 = load_scene(choice1, cfg.bbox, grid=grid, cache_dir=cache_dir)

    report_cb(0.30, "Computing spectral features for both dates")
    stack0, names = compute_feature_stack(scene0.bands)
    stack1, _ = compute_feature_stack(scene1.bands)
    valid0, valid1 = scene0.valid, scene1.valid
    if valid0.mean() < 0.2 or valid1.mean() < 0.2:
        raise RuntimeError(
            "Too much cloud or no-data in one of the dates for this area "
            f"({valid0.mean():.0%} and {valid1.mean():.0%} usable). "
            "Try a larger --max-cloud or a different date."
        )

    report_cb(0.38, "Building training labels from public mining data")
    labels = build_labels(
        grid,
        ndvi=stack0[..., names.index("NDVI")],
        ndwi=stack0[..., names.index("NDWI")],
        aoi_bbox_wgs84=cfg.bbox,
        external_dir=cfg.external_dir,
        concessions_path=cfg.concessions_path,
        extra_labels_path=hard_cases_path,
    )

    report_cb(0.45, "Preparing training rows (each date normalized on its own)")
    norm0 = normalize_per_scene(stack0, valid0)
    norm1 = normalize_per_scene(stack1, valid1)
    lab = labels.y
    rows, cols = np.nonzero((lab >= 0) & valid0 & valid1)
    if rows.size == 0:
        raise ValueError("No pixels are both labelled and clear on both dates.")
    y = lab[rows, cols].astype(int)
    X = np.concatenate([norm0[rows, cols], norm1[rows, cols]], axis=0)
    Y = np.concatenate([y, y])
    R = np.concatenate([rows, rows])
    C = np.concatenate([cols, cols])

    report_cb(0.55, "Comparing candidate models on a spatial hold-out (this takes a few minutes)")
    selection = fit_and_select(X, Y, R, C, names, seed=seed)
    best = selection.best
    report_cb(0.75, f"Best model: {best.name} (test F1 {best.f1_test:.3f})")

    detector = MiningDetector(best.model, names, normalize_per_scene=True)
    prob0 = detector.predict_proba_map(stack0, valid0)
    prob1 = detector.predict_proba_map(stack1, valid1)

    model_info = {
        "best": best.name,
        "candidates": [c.as_dict() for c in selection.candidates],
        "feature_importance": selection.feature_importance,
        "split": selection.split_kind,
        "training_rows": int(len(Y)),
        "label_counts": labels.counts,
    }
    if selection.random_split_f1:
        model_info["comparison"] = selection.random_split_f1

    data_sources = [
        f"Sentinel-2 L2A {choice0.item_id} ({choice0.source}), t0",
        f"Sentinel-2 L2A {choice1.item_id} ({choice1.source}), t1",
        f"Training labels: {labels.source}",
    ]
    if cfg.concessions_path:
        data_sources.append(f"Concessions: {cfg.concessions_path.name}")

    caveats = _build_caveats(labels, selection, choice0, choice1, cfg)
    caveats += labels.notes + selection.notes

    validation_points = None
    if validation_path is not None:
        from .validation import load_check_points

        validation_points = load_check_points(validation_path)
        data_sources.append(f"Hand-labelled check points: {Path(validation_path).name}")

    result = detect_and_write(
        stack0=stack0, stack1=stack1, prob0=prob0, prob1=prob1,
        transform=grid.transform, crs=grid.crs,
        concessions_utm=_load_concessions(cfg.concessions_path),
        cfg=cfg, bbox=cfg.bbox, data_sources=data_sources, caveats=caveats,
        model_info=model_info, progress=progress, validation_points=validation_points,
    )
    if review_chips > 0:
        from .validation import make_review_chips

        mask_t1 = _predicted_mask(prob0, prob1, cfg, grid.transform)
        csv_path = make_review_chips(cfg.out_dir, mask_t1, prob1, stack1, names,
                                     grid.transform, grid.crs, n=review_chips, seed=seed)
        report_cb(1.0, f"Review chips written: {csv_path}")
    result.metrics = {
        "best_model": best.name,
        "test_f1": round(best.f1_test, 4),
        "test_precision": round(best.precision_test, 4),
        "test_recall": round(best.recall_test, 4),
        "split": selection.split_kind,
        "runtime_s": round(time.time() - t_start, 1),
    }
    log.info("Live run finished in %.1f s", time.time() - t_start)
    return result


def _load_concessions(path: Path | None):
    """Read concession polygons, or return None when no file was given."""
    if path is None:
        return None
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def _build_caveats(labels, selection, choice0, choice1, cfg: Config) -> list[str]:
    """Plain-English limits that go into report.json. They are generated from the actual run."""
    caveats = [
        f"Training labels come from {labels.source.replace('_', ' ')}; mislabelled pixels cap the achievable accuracy.",
        "Scores on the spatial hold-out describe similar terrain only. Accuracy on other regions or on ambiguous land (roads, quarries, farms, burn scars) is not measured here.",
        "A near-perfect F1 on obvious cases does not imply accuracy on ambiguous land.",
        "The model was selected by test F1 on the same hold-out, so the reported F1 is slightly optimistic.",
        "Each alert is a satellite-detected change, not proof of illegal activity. Check the legal status against official concession records before acting.",
        f"Dates compared: {cfg.t0} vs {cfg.t1}. Clouds, shadows and sensor gaps are masked, so small patches can be missed.",
        f"Scenes used: {choice0.describe()}; {choice1.describe()}.",
    ]
    return caveats


def fetch_scene_report(
    bbox: tuple[float, float, float, float],
    date_str: str,
    max_cloud: float = 20.0,
    window_days: int = 15,
    cache_dir: str | Path = "data/cache",
    out_dir: str | Path = "outputs",
    progress: ProgressFn | None = None,
) -> dict:
    """Download, crop and cache one Sentinel-2 scene, then print a short report.

    This is a connectivity and data check. It does not run the detector.

    Returns:
        A dict with item_id, source, acquired, cloud_pct, size_mb, shape, valid_fraction
        and quicklook (path to a PNG).
    """
    from .data_sources import search_scenes
    from .preprocess import load_scene, target_grid

    report_cb = _report(progress)
    cfg_cache = Path(cache_dir)
    report_cb(0.05, f"Searching for a Sentinel-2 scene near {date_str} (cloud below {max_cloud:g}%)")
    choice = search_scenes(bbox, date_str, max_cloud=max_cloud, window_days=window_days)
    report_cb(0.3, f"Found {choice.describe()}")

    grid = target_grid(bbox)
    report_cb(0.4, f"Reading and cropping bands to a {grid.shape[1]} x {grid.shape[0]} px grid")
    scene = load_scene(choice, bbox, grid=grid, cache_dir=cfg_cache)

    cache_files = list(cfg_cache.glob("*.npz"))
    size_mb = sum(f.stat().st_size for f in cache_files) / 1e6 if cache_files else 0.0

    report_cb(0.85, "Writing a quicklook PNG")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"fetch_{choice.item_id}.png"
    rgb = _stretch_rgb(np.dstack([scene.bands["B04"], scene.bands["B03"], scene.bands["B02"]]))
    rgb[~scene.valid] = 0.2  # clouds and shadows drawn as grey
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(f"{choice.item_id}\n{choice.acquired.isoformat()}, cloud {choice.cloud_pct:.1f}%")
    ax.axis("off")
    fig.savefig(png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    result = {
        "item_id": choice.item_id,
        "source": choice.source,
        "acquired": choice.acquired.isoformat(),
        "cloud_pct": round(choice.cloud_pct, 2),
        "grid_px": list(grid.shape),
        "valid_fraction": round(scene.valid_fraction, 4),
        "cache_total_mb": round(size_mb, 2),
        "quicklook": str(png),
        "notes": scene.notes,
    }
    print()
    print("Fetch check")
    for k, v in result.items():
        print(f"  {k:<16}: {v}")
    report_cb(1.0, "Fetch complete")
    return result
