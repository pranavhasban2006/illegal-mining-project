"""Orchestrates O1-O5 end-to-end.

Two entry points:
  - run_demo(): fully synthetic, no CDSE credentials required. Use this now.
  - run_live(): real Sentinel-2 scenes via CDSE. Wire up once credentials exist.
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


def run_live(bbox, date_t0, date_t1, lease_path: str, out_dir: str = "outputs"):
    """Real CDSE pipeline. Requires CDSE_USERNAME / CDSE_PASSWORD env vars."""
    from . import auth, search, download, preprocess

    session = auth.CDSESession()

    items_t0 = search.search_scenes(bbox, date_t0, date_t0)
    items_t1 = search.search_scenes(bbox, date_t1, date_t1)
    if not items_t0 or not items_t1:
        raise RuntimeError("No suitable low-cloud scenes found for one of the dates.")

    paths_t0 = download.download_bands(items_t0[0], session, "data/raw/t0")
    paths_t1 = download.download_bands(items_t1[0], session, "data/raw/t1")

    bands_t0, transform, crs = preprocess.load_band_stack(paths_t0)
    bands_t1, _, _ = preprocess.load_band_stack(paths_t1)

    raise NotImplementedError(
        "Live pipeline needs labeled training pixels for this AOI before "
        "detection.MiningDetector.fit() can run -- plug in a labeled mask "
        "(manual annotation or an open mining-footprint dataset) here."
    )


if __name__ == "__main__":
    run_demo()
