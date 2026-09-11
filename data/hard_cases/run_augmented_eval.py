import sys, json
sys.path.insert(0, "../../src")
sys.path.insert(0, ".")

import geopandas as gpd
from mining_detector import real_run, preprocess, real_labels, detection, config, hard_case_labels
from hard_case_votes import VOTES

bands_t1_full, transform, crs = preprocess.load_band_stack(real_run._band_paths("../raw/test_scene"))
bands_t1, aoi_transform = preprocess.crop_to_bbox(bands_t1_full, transform, crs, config.PILOT_AOI_BBOX)

concessions = gpd.read_file("../leases/madre_de_dios_concessions.geojson")
base_labels = real_labels.generate_labels_from_concessions(bands_t1, aoi_transform, crs, concessions)

chips = json.load(open("chips_data.json"))
labeled_chips = hard_case_labels.load_votes_and_chips(VOTES, chips)
train_chips, test_chips = hard_case_labels.stratified_split(labeled_chips, train_fraction=0.5, seed=0)
print(f"Hard cases: {len(labeled_chips)} labeled (excl. unsure), "
      f"{len(train_chips)} train / {len(test_chips)} test")

print("\n=== BASELINE (concession labels only) ===")
detector_base = detection.MiningDetector()
report_base = detector_base.fit(bands_t1, base_labels, strategy="70:30", use_smote=True)
print(f"Selected model: {report_base['best_model_name']}")
fi_base = detector_base.feature_importances()
if fi_base:
    ranked = sorted(fi_base.items(), key=lambda kv: -kv[1])
    print("Feature importances (ranked):")
    for name, val in ranked:
        print(f"    {name:6s} {val:.3f}")

proba_base = detector_base.predict_proba_map(bands_t1)
metrics_base = hard_case_labels.evaluate_on_chips(proba_base, test_chips)
print(f"\nHeld-out hard-case test metrics (baseline model, n={metrics_base['n']}):")
print(f"    accuracy={metrics_base['accuracy']:.3f} precision={metrics_base['precision']:.3f} "
      f"recall={metrics_base['recall']:.3f} f1={metrics_base['f1']:.3f}")
print(f"    TP={metrics_base['tp']} FP={metrics_base['fp']} TN={metrics_base['tn']} FN={metrics_base['fn']}")

print("\n=== AUGMENTED (concession labels + hard-case training half) ===")
augmented_labels = hard_case_labels.apply_hard_case_labels(base_labels, train_chips, window=4)
detector_aug = detection.MiningDetector()
report_aug = detector_aug.fit(bands_t1, augmented_labels, strategy="70:30", use_smote=True)
print(f"Selected model: {report_aug['best_model_name']}")
fi_aug = detector_aug.feature_importances()
if fi_aug:
    ranked = sorted(fi_aug.items(), key=lambda kv: -kv[1])
    print("Feature importances (ranked):")
    for name, val in ranked:
        print(f"    {name:6s} {val:.3f}")

proba_aug = detector_aug.predict_proba_map(bands_t1)
metrics_aug = hard_case_labels.evaluate_on_chips(proba_aug, test_chips)
print(f"\nHeld-out hard-case test metrics (augmented model, SAME {metrics_aug['n']} held-out chips):")
print(f"    accuracy={metrics_aug['accuracy']:.3f} precision={metrics_aug['precision']:.3f} "
      f"recall={metrics_aug['recall']:.3f} f1={metrics_aug['f1']:.3f}")
print(f"    TP={metrics_aug['tp']} FP={metrics_aug['fp']} TN={metrics_aug['tn']} FN={metrics_aug['fn']}")

json.dump({
    "n_train_hard": len(train_chips), "n_test_hard": len(test_chips),
    "baseline": metrics_base, "augmented": metrics_aug,
    "baseline_feature_importances": fi_base, "augmented_feature_importances": fi_aug,
}, open("augmented_eval_results.json", "w"), indent=2)
print("\nWrote augmented_eval_results.json")
