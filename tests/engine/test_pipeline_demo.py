"""End-to-end test of the offline demo against its known answer."""

from __future__ import annotations

import json

import numpy as np

from mining_detector.pipeline import run_demo
from mining_detector.synthetic import make_synthetic_scene


def test_synthetic_scene_has_known_truth():
    scene = make_synthetic_scene(seed=1)
    assert scene.truth_new_mask.sum() > 0
    # New sites must not overlap the licensed mine, and the licence must cover it.
    assert not (scene.truth_new_mask & scene.licensed_mask).any()
    assert len(scene.new_site_boxes) == 2
    assert len(scene.concessions) == 1


def test_demo_finds_both_new_sites_and_writes_contract_files(tmp_path):
    result = run_demo(out_dir=tmp_path)

    assert result.alerts_path.exists()
    assert result.report_path.exists()
    assert (tmp_path / "change_map.png").exists()
    assert (tmp_path / "probability_t1.png").exists()

    fc = json.loads(result.alerts_path.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    # Both sites are far from the licence (more than the 500 m buffer), so both are illegal.
    assert all(f["properties"]["legal_status"] == "illegal" for f in fc["features"])

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    for key in ("aoi_bbox", "t0", "t1", "n_alerts", "total_new_disturbance_ha",
                "by_legal_status", "model", "data_sources", "caveats"):
        assert key in report
    assert report["n_alerts"] == 2
    assert report["by_legal_status"]["legal"] == 0


def test_demo_pixel_metrics_are_high_on_known_answer(tmp_path):
    result = run_demo(out_dir=tmp_path)
    assert result.metrics["pixel_f1"] > 0.95
    assert result.metrics["found_alerts"] == result.metrics["expected_alerts"]


def test_demo_is_deterministic(tmp_path):
    a = run_demo(out_dir=tmp_path / "a").summary
    b = run_demo(out_dir=tmp_path / "b").summary
    a.pop("runtime_s")
    b.pop("runtime_s")
    assert a == b
