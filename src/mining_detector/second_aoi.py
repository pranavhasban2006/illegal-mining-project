"""Multi-region generalization test (addresses the SAR paper's and
EuroMineNet's future-work item: "further work on multi-region
generalisation to reduce the accuracy gap observed between training and
transfer domains"). Both papers are cited in Project Review 1.

A second, genuinely distinct mining concession cluster (~45km west of the
Madre de Dios / La Pampa pilot AOI, found by clustering the real RAISG
concessions dataset rather than an imprecise news-article coordinate) is
used as a held-out transfer-test region -- 420 real concessions once
re-clipped specifically for this AOI (see AOI2_CONCESSIONS_PATH; the
AOI1-only concessions extract does NOT extend here). Two evaluations are
compared:
  1. TRANSFER: the classifier trained on AOI1 (Madre de Dios/La Pampa,
     real_run.py) applied directly to AOI2's imagery, no retraining.
  2. IN-DOMAIN: a fresh classifier trained on AOI2's own concession data,
     evaluated on AOI2 -- the ceiling this specific AOI's data supports.
The gap between (1) and (2) is the actual measured cross-region
generalization gap, not an assumption.
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score

from . import preprocess, real_labels, real_run, detection, indices

AOI2_BBOX = (-70.65, -13.10, -70.45, -13.00)  # ~45km west of the Madre de Dios pilot AOI
AOI2_T0_DIR = "data/raw/aoi2_t0_2025"  # 2025-07-27
AOI2_T1_DIR = "data/raw/aoi2_t1_2026"  # 2026-07-27
AOI1_CONCESSIONS_PATH = "data/leases/madre_de_dios_concessions.geojson"  # buffered around AOI1 only
AOI2_CONCESSIONS_PATH = "data/leases/aoi2_concessions.geojson"  # buffered around AOI2 only -- separate
# extract: AOI1's concessions file does NOT extend to AOI2 (45km away), so
# reusing it here would (and initially did) produce zero positive labels.


def _evaluate_against_labels(proba: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """labels: -1=unlabeled/excluded, 0/1 ground truth -- same convention as
    real_labels.py. Only labeled pixels are scored."""
    mask = labels != -1
    y_true = labels[mask]
    y_pred = (proba[mask] >= threshold).astype(int)
    return {
        "n": int(mask.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "iou": jaccard_score(y_true, y_pred, zero_division=0),
    }


def run():
    print("[AOI2] Loading second AOI imagery (real concession cluster, ~45km "
          "west of the Madre de Dios pilot AOI)...")
    bands_t1_full, transform, crs = preprocess.load_band_stack(real_run._band_paths(AOI2_T1_DIR))
    bands_t1_aoi2, aoi2_transform = preprocess.crop_to_bbox(bands_t1_full, transform, crs, AOI2_BBOX)
    print(f"[AOI2] Shape: {bands_t1_aoi2.shape}")

    concessions_aoi2 = gpd.read_file(AOI2_CONCESSIONS_PATH)
    print(f"[AOI2] Loaded {len(concessions_aoi2)} real RAISG concessions specific to AOI2 "
          f"({len(real_labels.active_concessions(concessions_aoi2))} active/titled)")

    aoi2_labels = real_labels.generate_labels_from_concessions(
        bands_t1_aoi2, aoi2_transform, crs, concessions_aoi2
    )
    print(f"[AOI2] Label summary: mining={int((aoi2_labels == 1).sum())}px, "
          f"not_mining={int((aoi2_labels == 0).sum())}px")

    # --- Evaluation 1: TRANSFER (train on AOI1, no retraining, test on AOI2) ---
    print("\n[Transfer] Training on AOI1 (Madre de Dios)...")
    bands_t1_aoi1_full, transform1, crs1 = preprocess.load_band_stack(real_run._band_paths(real_run.T1_DIR))
    import mining_detector.config as config
    bands_t1_aoi1, aoi1_transform = preprocess.crop_to_bbox(bands_t1_aoi1_full, transform1, crs1, config.PILOT_AOI_BBOX)
    concessions_aoi1 = gpd.read_file(AOI1_CONCESSIONS_PATH)
    aoi1_labels = real_labels.generate_labels_from_concessions(bands_t1_aoi1, aoi1_transform, crs1, concessions_aoi1)

    detector_transfer = detection.MiningDetector()
    detector_transfer.fit(bands_t1_aoi1, aoi1_labels, strategy="70:30", use_smote=True)
    proba_transfer = detector_transfer.predict_proba_map(bands_t1_aoi2)
    transfer_metrics = _evaluate_against_labels(proba_transfer, aoi2_labels)
    print(f"[Transfer] AOI1-trained model on AOI2: {transfer_metrics}")

    # --- Evaluation 2: IN-DOMAIN (train AND test on AOI2) ---
    print("\n[In-domain] Training fresh model on AOI2's own data...")
    detector_indomain = detection.MiningDetector()
    report = detector_indomain.fit(bands_t1_aoi2, aoi2_labels, strategy="70:30", use_smote=True)
    indomain_metrics = report["results"][report["best_model_name"]]["test"]
    print(f"[In-domain] Best model ({report['best_model_name']}) on its own held-out AOI2 split: {indomain_metrics}")

    print(f"\n=== Generalization gap (in-domain F1 - transfer F1) ===")
    gap = indomain_metrics["f1"] - transfer_metrics["f1"]
    print(f"In-domain F1: {indomain_metrics['f1']:.3f}  |  Transfer F1: {transfer_metrics['f1']:.3f}  |  Gap: {gap:.3f}")

    return {"transfer": transfer_metrics, "in_domain": indomain_metrics, "gap_f1": gap}


if __name__ == "__main__":
    run()
