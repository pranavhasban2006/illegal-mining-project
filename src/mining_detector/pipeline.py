"""Orchestrates O1-O5 end-to-end.

Two entry points:
  - run_demo(): fully synthetic, no CDSE credentials required. Good for
    demoing the architecture without any setup.
  - run_live(): the real pipeline against Sentinel-2 imagery over the Peru
    pilot AOI (see config.py). Delegates to real_run.py, which implements
    the full flow: CDSE download, real RAISG-concession-based labels
    (real_labels.py) corrected with hand-labeled hard cases
    (hard_case_labels.py), bi-temporal change detection, and 3-tier
    legal/disputed/illegal classification (licensing.py).
"""

from __future__ import annotations

import os
import geopandas as gpd

from . import demo_data, detection, change_detection, licensing, alerts


def run_demo(out_dir: str = "outputs", lease_path: str = "data/leases/demo_leases.geojson"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(lease_path), exist_ok=True)

    scene = demo_data.generate_bitemporal_scene()
    demo_data.make_lease_boundaries(lease_path)

    detector = detection.MiningDetector()
    report = detector.fit(
        scene["t1"], scene["ground_truth_new_illegal"] | _pre_existing_mask(),
        strategy="70:30", use_smote=True,
    )
    print(f"[O2] Split: {report['split_scheme']} holdout, SMOTE={report['used_smote']}")
    print("[O2] Algorithm comparison (test set):")
    for name, res in report["results"].items():
        flag = " OVERFIT" if res["overfit_flag"] else ""
        print(f"    {name:28s} f1={res['test']['f1']:.3f} "
              f"iou={res['test']['iou']:.3f} "
              f"precision={res['test']['precision']:.3f} "
              f"recall={res['test']['recall']:.3f}{flag}")
    print(f"[O2] Selected best model: {report['best_model_name']}")
    fi = detector.feature_importances()
    if fi:
        print(f"[O2] Feature importances: {fi}")

    proba_t0 = detector.predict_proba_map(scene["t0"])
    proba_t1 = detector.predict_proba_map(scene["t1"])

    new_mask = change_detection.new_disturbance_mask(proba_t0, proba_t1)
    change_conf = change_detection.change_confidence(proba_t0, proba_t1)
    print(f"[O3] New-disturbance pixels detected: {int(new_mask.sum())}")

    lease_gdf = gpd.read_file(lease_path)
    licensed_mask = licensing.rasterize_leases(
        lease_gdf, new_mask.shape, scene["transform"], scene["crs"]
    )
    suspicious_mask = licensing.suspicious_area_mask(new_mask, licensed_mask)
    confidence_map = licensing.suspicious_confidence(suspicious_mask, proba_t1, change_conf)
    print(f"[O4] Suspicious (unlicensed new-disturbance) pixels: {int(suspicious_mask.sum())}")

    alert_gdf = alerts.vectorize_alerts(
        suspicious_mask, confidence_map, scene["transform"], scene["crs"],
        observation_date="2026-demo-t1",
    )
    out_path = os.path.join(out_dir, "alerts.geojson")
    alerts.write_geojson(alert_gdf, out_path)
    print(f"[O5] Wrote {len(alert_gdf)} alert(s) to {out_path}")
    if len(alert_gdf):
        print(alert_gdf[["alert_id", "confidence", "area_ha"]].to_string(index=False))

    return {
        "model_selection_report": report,
        "n_new_disturbance_px": int(new_mask.sum()),
        "n_suspicious_px": int(suspicious_mask.sum()),
        "alerts_path": out_path,
        "n_alerts": len(alert_gdf),
    }


def _pre_existing_mask():
    """Ground-truth for the pre-existing licensed mine so the classifier also
    learns what generic bare-soil/mining spectral signature looks like
    (not just the two new illegal sites)."""
    import numpy as np
    m = np.zeros((demo_data.HEIGHT, demo_data.WIDTH), dtype=np.uint8)
    m[25:75, 25:75] = 1
    return m


def run_live(out_dir: str = "outputs"):
    """Real Sentinel-2 pipeline over the Peru pilot AOI. Requires:
      - CDSE_USERNAME/CDSE_PASSWORD or CDSE_S3_ACCESS_KEY/CDSE_S3_SECRET_KEY
        in .env (see README) -- only needed if data/raw/ scenes aren't
        already downloaded (download.py caches to disk).
      - data/leases/madre_de_dios_concessions.geojson (RAISG concessions,
        see README for how this was obtained).

    See real_run.py for the full implementation; this is a thin wrapper so
    `pipeline` stays the single entry-point module for both demo and live runs.
    """
    from . import real_run
    return real_run.run(out_dir=out_dir)


if __name__ == "__main__":
    run_demo()
