"""End-to-end O1-O5 run against real Sentinel-2 data for the Peru pilot AOI.

Assumes bands for t0 and t1 have already been downloaded (see download.py)
into data/raw/t0_scene_2025 and data/raw/test_scene. Training labels start
from data/leases/madre_de_dios_concessions.geojson (a real RAISG mining-
concessions extract -- see real_labels.py for why concessions make a
genuine ground-truth source), then are corrected with hand-labeled hard
cases from data/hard_cases/ when available (see hard_case_labels.py):
validation showed the concession-only model over-flags moderate vegetation
loss as mining, and folding in real human labels for exactly that confusion
measurably fixed it (held-out F1 0.833 -> 0.909, see
data/hard_cases/validation_summary.md).
"""

from __future__ import annotations

import json
import os
import geopandas as gpd

from . import config, preprocess, real_labels, weak_labels, hard_case_labels, detection, change_detection, licensing, alerts

T0_DIR = "data/raw/t0_scene_2025"  # 2025-07-27 -- same calendar day, prior year
T1_DIR = "data/raw/test_scene"     # 2026-07-27 -- see config.py for why same-month matters
CONCESSIONS_PATH = "data/leases/madre_de_dios_concessions.geojson"
HARD_CASES_DIR = "data/hard_cases"


def _band_paths(scene_dir: str) -> dict:
    return {
        "B02": os.path.join(scene_dir, "B02_10m.jp2"),
        "B03": os.path.join(scene_dir, "B03_10m.jp2"),
        "B04": os.path.join(scene_dir, "B04_10m.jp2"),
        "B08": os.path.join(scene_dir, "B08_10m.jp2"),
        "B11": os.path.join(scene_dir, "B11_20m.jp2"),
        "B12": os.path.join(scene_dir, "B12_20m.jp2"),
    }


def run(out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)

    print("[O1] Loading and cropping t0/t1 scenes to pilot AOI...")
    bands_t0_full, transform, crs = preprocess.load_band_stack(_band_paths(T0_DIR))
    bands_t1_full, _, _ = preprocess.load_band_stack(_band_paths(T1_DIR))

    bands_t0, aoi_transform = preprocess.crop_to_bbox(bands_t0_full, transform, crs, config.PILOT_AOI_BBOX)
    bands_t1, _ = preprocess.crop_to_bbox(bands_t1_full, transform, crs, config.PILOT_AOI_BBOX)
    print(f"[O1] AOI shape: {bands_t1.shape}")

    concessions = gpd.read_file(CONCESSIONS_PATH)
    print(f"[O1] Loaded {len(concessions)} RAISG mining concessions "
          f"({len(real_labels.active_concessions(concessions))} active/titled)")

    print("[O2] Generating real labels from RAISG concessions (see real_labels.py)...")
    labels = real_labels.generate_labels_from_concessions(
        bands_t1, aoi_transform, crs, concessions
    )
    print(f"[O2] Label summary: {weak_labels.label_summary(labels)}")

    chips_path = os.path.join(HARD_CASES_DIR, "chips_data.json")
    votes_path = os.path.join(HARD_CASES_DIR, "votes.json")
    if os.path.exists(chips_path) and os.path.exists(votes_path):
        chips = json.load(open(chips_path))
        votes = json.load(open(votes_path))
        labeled_chips = hard_case_labels.load_votes_and_chips(votes, chips)
        labels = hard_case_labels.apply_hard_case_labels(labels, labeled_chips, window=4)
        print(f"[O2] Folded in {len(labeled_chips)} hand-labeled hard cases "
              f"from {HARD_CASES_DIR} (see validation_summary.md)")

    detector = detection.MiningDetector()
    report = detector.fit(bands_t1, labels, strategy="70:30", use_smote=True)
    print(f"[O2] Split: {report['split_scheme']} holdout, SMOTE={report['used_smote']}")
    for name, res in report["results"].items():
        flag = " OVERFIT" if res["overfit_flag"] else ""
        print(f"    {name:28s} f1={res['test']['f1']:.3f} iou={res['test']['iou']:.3f}{flag}")
    print(f"[O2] Selected best model: {report['best_model_name']}")

    proba_t0 = detector.predict_proba_map(bands_t0)
    proba_t1 = detector.predict_proba_map(bands_t1)

    print("[O3] Running change detection...")
    new_mask = change_detection.new_disturbance_mask(proba_t0, proba_t1)
    change_conf = change_detection.change_confidence(proba_t0, proba_t1)
    print(f"[O3] New-disturbance pixels: {int(new_mask.sum())} "
          f"(~{new_mask.sum() * 100 / 10_000:.1f} ha)")

    print("[O4] Classifying legal status against RAISG concessions "
          "(legal / disputed / illegal)...")
    active = real_labels.active_concessions(concessions)
    active_mask = licensing.rasterize_leases(active, new_mask.shape, aoi_transform, crs)
    any_concession_mask = licensing.rasterize_leases(concessions, new_mask.shape, aoi_transform, crs)
    legal_status = licensing.classify_legal_status(new_mask, active_mask, any_concession_mask)
    confidence_map = licensing.status_confidence(legal_status, proba_t1, change_conf)

    from .licensing import LegalStatus
    n_legal = int((legal_status == LegalStatus.LEGAL).sum())
    n_disputed = int((legal_status == LegalStatus.DISPUTED).sum())
    n_illegal = int((legal_status == LegalStatus.ILLEGAL).sum())
    print(f"[O4] New-disturbance breakdown: legal={n_legal}px (~{n_legal/100:.1f}ha), "
          f"disputed={n_disputed}px (~{n_disputed/100:.1f}ha), "
          f"illegal={n_illegal}px (~{n_illegal/100:.1f}ha)")

    print("[O5] Exporting alerts (disputed + illegal only; legal excluded)...")
    alert_gdf = alerts.vectorize_status_alerts(
        legal_status, confidence_map, aoi_transform, crs,
        observation_date=config.DEFAULT_DATE_T1, min_area_m2=2000.0,
    )
    out_path = os.path.join(out_dir, "real_alerts.geojson")
    alerts.write_geojson(alert_gdf, out_path)
    print(f"[O5] Wrote {len(alert_gdf)} alert(s) to {out_path}")
    if len(alert_gdf):
        print(alert_gdf["legal_status"].value_counts().to_string())
        print(alert_gdf[["alert_id", "legal_status", "confidence", "area_ha"]].head(10).to_string(index=False))

    return alert_gdf


if __name__ == "__main__":
    run()
