"""Check the detector against people's labels, and prepare chips for labelling.

Two things live here:
1. evaluate_points: compares the t1 alert mask with hand-labelled points (label 1 = mining,
   0 = not mining, "unsure" rows are ignored). Results go into report.json under "validation".
2. make_review_chips: exports small stratified image chips and a CSV template, so a person
   can label the hardest cases. Terminal B's app uses these files.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from affine import Affine  # noqa: E402
from pyproj import Transformer  # noqa: E402
from scipy import ndimage  # noqa: E402

log = logging.getLogger(__name__)

UNSURE = "unsure"


def load_check_points(path: str | Path) -> gpd.GeoDataFrame:
    """Read hand-labelled points from CSV (lon, lat, label) or GeoJSON (label property).

    Returns:
        GeoDataFrame in EPSG:4326 with a 'label' column of 0 or 1. Rows marked unsure, or
        with any other value, are removed. The number removed is in gdf.attrs["dropped_unsure"].

    Raises:
        ValueError: if required columns are missing.
    """
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if not {"lon", "lat", "label"}.issubset(df.columns):
            raise ValueError("Check-point CSV needs columns: lon, lat, label")
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    else:
        gdf = gpd.read_file(path)
        if "label" not in gdf.columns:
            raise ValueError("Check-point file needs a 'label' property (0, 1 or unsure).")
        gdf = gdf.to_crs("EPSG:4326")

    raw = gdf["label"].astype(str).str.strip().str.lower()
    dropped = int((~raw.isin(["0", "1"])).sum())
    gdf = gdf.assign(label=raw[raw.isin(["0", "1"])].astype(int)).dropna(subset=["label"])
    gdf = gdf.loc[gdf.geometry.notna()]
    gdf.attrs["dropped_unsure"] = dropped
    return gdf


def evaluate_points(
    points: gpd.GeoDataFrame,
    mask: np.ndarray,
    transform: Affine,
    crs: str,
) -> dict:
    """Confusion matrix and precision, recall and F1 of the mask against labelled points.

    A point counts as predicted-mining when the pixel under it is True in the mask.
    Points that fall outside the mask grid are counted but not scored.
    """
    to_grid = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    tp = fp = tn = fn = outside = 0
    h, w = mask.shape
    for lon, lat, lab in zip(points.geometry.x, points.geometry.y, points["label"]):
        x, y = to_grid.transform(lon, lat)
        col = int(np.floor((x - transform.c) / transform.a))
        row = int(np.floor((y - transform.f) / transform.e))
        if not (0 <= row < h and 0 <= col < w):
            outside += 1
            continue
        pred = bool(mask[row, col])
        if pred and lab == 1:
            tp += 1
        elif pred and lab == 0:
            fp += 1
        elif not pred and lab == 0:
            tn += 1
        else:
            fn += 1

    scored = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_points": int(len(points)),
        "n_scored": int(scored),
        "n_outside_grid": int(outside),
        "n_unsure_dropped": int(points.attrs.get("dropped_unsure", 0)),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "caveat": (
            "Small hand-labelled sample: treat these numbers as indicative, not as a "
            "reliable accuracy estimate."
            if scored < 100 else "Hand-labelled sample evaluation."
        ),
    }


def write_validation_into_report(report_path: str | Path, validation: dict) -> None:
    """Add (or replace) the 'validation' section of an existing report.json."""
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["validation"] = validation
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---------- hard-case hold-out evaluation ----------

def split_hard_cases(path: str | Path, out_dir: str | Path, seed: int = 42) -> tuple[Path, Path]:
    """Split hand labels into two random halves, A and B, written as CSV files.

    Returns:
        (path to half A, path to half B).
    """
    import pandas as pd

    df = pd.read_csv(path)
    if not {"lon", "lat", "label"}.issubset(df.columns):
        raise ValueError("Hard-case CSV needs columns: lon, lat, label")
    df = df[df["label"].astype(str).isin(["0", "1"])].reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("A hold-out evaluation needs at least 4 hand-labelled points.")
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    half = len(shuffled) // 2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a_path = out / "hard_cases_half_A.csv"
    b_path = out / "hard_cases_half_B.csv"
    shuffled.iloc[:half].to_csv(a_path, index=False)
    shuffled.iloc[half:].to_csv(b_path, index=False)
    return a_path, b_path


def holdout_gain(run_live_fn, hard_cases_path: str | Path, out_dir: str | Path, **run_kwargs) -> dict:
    """Measure whether hand labels improve the detector on cases they did not train on.

    Protocol:
        1. Split the hand labels into halves A and B.
        2. Baseline: run with no hand labels, score against half B.
        3. Trained: run with half A folded into training, score against half B.
        The difference in F1 on half B is the measured gain. Half B is never used in training.

    Args:
        run_live_fn: the pipeline's run_live function (passed in to avoid a circular import).
        hard_cases_path: the full hand-label CSV.
        out_dir: where the two runs and the halves are written.
        **run_kwargs: other arguments for run_live (aoi_bbox, t0_date, t1_date, ...).

    Returns:
        dict with baseline_f1, trained_f1, gain, and the number of points in each half.
    """
    import pandas as pd

    out = Path(out_dir)
    a_path, b_path = split_hard_cases(hard_cases_path, out)

    base = run_live_fn(out_dir=out / "baseline", validation_path=b_path, **run_kwargs)
    trained = run_live_fn(out_dir=out / "trained", hard_cases_path=a_path,
                          validation_path=b_path, **run_kwargs)

    base_v = json.loads(base.report_path.read_text(encoding="utf-8")).get("validation", {})
    trained_v = json.loads(trained.report_path.read_text(encoding="utf-8")).get("validation", {})
    base_f1 = float(base_v.get("f1", 0.0))
    trained_f1 = float(trained_v.get("f1", 0.0))
    result = {
        "half_a_points": len(pd.read_csv(a_path)),
        "half_b_points": len(pd.read_csv(b_path)),
        "baseline_f1_on_half_b": round(base_f1, 4),
        "trained_f1_on_half_b": round(trained_f1, 4),
        "gain": round(trained_f1 - base_f1, 4),
        "caveat": "Measured on one random split of a small hand-labelled set; treat as indicative.",
    }
    (out / "hard_case_holdout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------- review chips ----------

CHIP_CATEGORIES = ("near_boundary", "uncertain_probability", "bare_far_from_mine", "mid_ndvi")


def _candidate_pixels(category: str, mask, prob1, stack1, names, far_m, pixel_m) -> np.ndarray:
    """Return a boolean mask of pixels that qualify for a category."""
    ndvi = stack1[..., names.index("NDVI")]
    bsi = stack1[..., names.index("BSI")]
    if category == "near_boundary":
        # Pixels within 2 px of an alert edge, on either side.
        edge = ndimage.binary_dilation(mask, iterations=2) ^ ndimage.binary_erosion(mask, iterations=2)
        return edge
    if category == "uncertain_probability":
        return (prob1 >= 0.4) & (prob1 <= 0.6)
    if category == "bare_far_from_mine":
        dist_m = ndimage.distance_transform_edt(~mask) * pixel_m if mask.any() else np.full(mask.shape, np.inf)
        return (bsi > 0.05) & (prob1 < 0.3) & (dist_m > far_m)
    if category == "mid_ndvi":
        return (ndvi > 0.3) & (ndvi < 0.5)
    raise ValueError(f"Unknown category {category}")


def make_review_chips(
    out_dir: str | Path,
    mask: np.ndarray,
    prob1: np.ndarray,
    stack1: np.ndarray,
    names: list[str],
    transform: Affine,
    crs: str,
    n: int = 40,
    chip_px: int = 64,
    seed: int = 42,
) -> Path:
    """Export up to n stratified review chips (PNG) and a labelling CSV template.

    The n chips are split evenly across the four categories. Each category draws from
    pixels that qualify for it, and where a category has fewer candidates, fewer chips
    are made for it (the CSV records the category for each chip).

    Returns:
        Path to the CSV template (chips/review_template.csv).
    """
    out = Path(out_dir) / "chips"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    pixel_m = abs(transform.a)
    per_cat = max(1, n // len(CHIP_CATEGORIES))
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    half = chip_px // 2
    rgb_full = np.clip(stack1[..., [2, 1, 0]], 0, 1) ** 0.6

    rows_out = []
    chip_no = 0
    for cat in CHIP_CATEGORIES:
        cand = _candidate_pixels(cat, mask, prob1, stack1, names, far_m=1000.0, pixel_m=pixel_m)
        ys, xs = np.nonzero(cand[half:-half, half:-half])
        if ys.size == 0:
            log.info("No candidate pixels for chip category %s", cat)
            continue
        pick = rng.choice(ys.size, size=min(per_cat, ys.size), replace=False)
        for k in pick:
            r, c = ys[k] + half, xs[k] + half
            chip_no += 1
            name = f"chip_{chip_no:03d}_{cat}.png"
            chip = rgb_full[r - half:r + half, c - half:c + half]
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(chip)
            ax.plot([half], [half], marker="+", color="red", markersize=10)
            ax.axis("off")
            fig.savefig(out / name, dpi=100, bbox_inches="tight")
            plt.close(fig)
            x = transform.c + (c + 0.5) * transform.a
            y = transform.f + (r + 0.5) * transform.e
            lon, lat = to_ll.transform(x, y)
            rows_out.append({
                "chip_id": chip_no, "category": cat, "file": name,
                "lon": round(lon, 6), "lat": round(lat, 6),
                "label": "", "notes": "",
            })

    csv_path = out / "review_template.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chip_id", "category", "file", "lon", "lat", "label", "notes"])
        writer.writeheader()
        writer.writerows(rows_out)
    log.info("Wrote %d review chips to %s", len(rows_out), out)
    return csv_path
