"""End-to-end O1-O5 run against real Sentinel-2 data for the Peru pilot AOI.

Assumes bands for t0 and t1 have already been downloaded (see download.py)
into data/raw/t0_scene and data/raw/t1_scene. No lease-boundary data exists
publicly for this AOI yet, so licensing.py runs with an empty lease set --
every detected new-disturbance polygon is flagged suspicious pending real
lease data (see README for how to plug that in).
"""

from __future__ import annotations

import os
import numpy as np
import geopandas as gpd

from . import config, preprocess, weak_labels, detection, change_detection, licensing, alerts

T0_DIR = "data/raw/t0_scene_2025"  # 2025-07-27 -- same calendar day, prior year
T1_DIR = "data/raw/test_scene"     # 2026-07-27 -- see config.py for why same-month matters


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

    print("[O2] Generating weak labels from t1 (see weak_labels.py)...")
    labels = weak_labels.generate_weak_labels(bands_t1)
    print(f"[O2] Label summary: {weak_labels.label_summary(labels)}")

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

    print("[O4] Cross-referencing against licensed leases (none available for this AOI yet)...")
    empty_leases = gpd.GeoDataFrame(geometry=[], crs=crs)
    licensed_mask = licensing.rasterize_leases(empty_leases, new_mask.shape, aoi_transform, crs)
    suspicious_mask = licensing.suspicious_area_mask(new_mask, licensed_mask)
    confidence_map = licensing.suspicious_confidence(suspicious_mask, proba_t1, change_conf)
    print(f"[O4] Suspicious pixels: {int(suspicious_mask.sum())}")

    print("[O5] Exporting alerts...")
    alert_gdf = alerts.vectorize_alerts(
        suspicious_mask, confidence_map, aoi_transform, crs,
        observation_date=config.DEFAULT_DATE_T1, min_area_m2=2000.0,
    )
    out_path = os.path.join(out_dir, "real_alerts.geojson")
    alerts.write_geojson(alert_gdf, out_path)
    print(f"[O5] Wrote {len(alert_gdf)} alert(s) to {out_path}")
    if len(alert_gdf):
        print(alert_gdf[["alert_id", "confidence", "area_ha"]].head(10).to_string(index=False))

    return alert_gdf


if __name__ == "__main__":
    run()
