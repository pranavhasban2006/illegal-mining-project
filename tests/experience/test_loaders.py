"""Tests for the app's data loading and summary logic (app/lib/loaders.py).

These use tiny synthetic runs whose correct answers are worked out by hand, so a
wrong total shows up as a clear failure.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))

from lib.loaders import (  # noqa: E402
    alerts_to_csv,
    choose_data_folder,
    filter_alerts,
    find_runs,
    load_alerts_table,
    load_run,
    summarize,
)
from lib.paths import SAMPLE_RUN_DIR  # noqa: E402


def _feature(i, status, conf, area):
    """One square-ish polygon feature with the contract's properties."""
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        "properties": {"id": i, "legal_status": status, "confidence": conf,
                       "area_ha": area, "detected_on": "2026-06-01"},
    }


# Known answer: 4 alerts. illegal = 10 + 5 = 15 ha, disputed = 2 ha, legal = 0 ha.
KNOWN = [
    _feature(1, "illegal", 0.9, 10.0),
    _feature(2, "illegal", 0.6, 5.0),
    _feature(3, "disputed", 0.4, 2.0),
    _feature(4, "illegal", 0.95, 0.5),
]


def _write_run(folder: Path, features, n_alerts=None, mtime_offset=0):
    """Write alerts.geojson and report.json into folder, as the engine would."""
    folder.mkdir(parents=True, exist_ok=True)
    geo = {"type": "FeatureCollection", "features": features}
    report = {
        "aoi_bbox": [-71.0, -13.0, -70.8, -12.8],
        "t0": "2025-06-01", "t1": "2026-06-01",
        "n_alerts": len(features) if n_alerts is None else n_alerts,
        "caveats": ["test caveat"],
        "data_sources": ["unit test"],
    }
    (folder / "alerts.geojson").write_text(json.dumps(geo), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    if mtime_offset:
        import os
        t = (folder / "alerts.geojson").stat().st_mtime + mtime_offset
        os.utime(folder / "alerts.geojson", (t, t))
    return folder


def test_summarize_known_answer(tmp_path):
    run = load_run(_write_run(tmp_path / "run", KNOWN))
    stats = summarize(run.alerts)
    assert stats["n_alerts"] == 4
    assert stats["total_ha"] == pytest.approx(17.5)
    assert stats["by_status_ha"] == {"legal": 0.0, "disputed": 2.0, "illegal": 15.5}
    assert stats["by_status_count"] == {"legal": 0, "disputed": 1, "illegal": 3}


def test_summarize_empty_table_has_every_status():
    stats = summarize(load_alerts_table({"features": []}))
    assert stats["n_alerts"] == 0
    assert stats["total_ha"] == 0.0
    assert set(stats["by_status_ha"]) == {"legal", "disputed", "illegal"}


def test_unknown_status_is_rejected():
    bad = [_feature(1, "maybe", 0.5, 1.0)]
    with pytest.raises(ValueError, match="Unknown legal_status"):
        load_alerts_table({"features": bad})


def test_filter_by_status_and_confidence(tmp_path):
    run = load_run(_write_run(tmp_path / "run", KNOWN))
    # Illegal with confidence >= 0.8: ids 1 and 4 only.
    kept = filter_alerts(run.alerts, ["illegal"], 0.8)
    assert list(kept["id"]) == [1, 4]
    # Disputed is excluded when not selected.
    assert "disputed" not in set(filter_alerts(run.alerts, ["illegal", "legal"], 0.0)["legal_status"])


def test_filter_threshold_is_inclusive(tmp_path):
    run = load_run(_write_run(tmp_path / "run", KNOWN))
    # Id 2 has confidence exactly 0.6, so it must be kept. Id 3 (0.4) is dropped.
    kept = filter_alerts(run.alerts, ["illegal", "disputed"], 0.6)
    assert sorted(kept["id"]) == [1, 2, 4]


def test_load_run_warns_when_counts_disagree(tmp_path):
    run = load_run(_write_run(tmp_path / "run", KNOWN, n_alerts=7))
    assert any("says 7 alerts" in w for w in run.warnings)


def test_load_run_missing_file_is_plain_message(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="missing alerts.geojson"):
        load_run(folder)


def test_csv_export_round_trips(tmp_path):
    run = load_run(_write_run(tmp_path / "run", KNOWN))
    text = alerts_to_csv(run.alerts).decode("utf-8")
    back = pd.read_csv(pd.io.common.StringIO(text))
    assert list(back.columns) == ["id", "legal_status", "confidence", "area_ha", "detected_on"]
    assert back["area_ha"].sum() == pytest.approx(17.5)


def test_find_runs_newest_first(tmp_path):
    old = _write_run(tmp_path / "old", KNOWN, mtime_offset=-100)
    new = _write_run(tmp_path / "new", KNOWN)
    (tmp_path / "not_a_run").mkdir()
    assert find_runs(tmp_path) == [new, old]


def test_find_runs_missing_folder_returns_empty(tmp_path):
    assert find_runs(tmp_path / "does_not_exist") == []


def test_choose_prefers_real_output_over_sample(tmp_path):
    real = _write_run(tmp_path / "outputs" / "run1", KNOWN)
    folder, is_sample = choose_data_folder(tmp_path / "outputs", SAMPLE_RUN_DIR)
    assert folder == real and is_sample is False


def test_choose_falls_back_to_sample_when_outputs_empty(tmp_path):
    folder, is_sample = choose_data_folder(tmp_path / "outputs", SAMPLE_RUN_DIR)
    assert folder == SAMPLE_RUN_DIR and is_sample is True


def test_choose_returns_none_when_nothing_exists(tmp_path):
    assert choose_data_folder(tmp_path / "outputs", tmp_path / "nope") == (None, False)


def test_generated_sample_run_matches_its_known_answer():
    """The bundled sample run must load and total exactly what make_fixtures.py intended."""
    if not (SAMPLE_RUN_DIR / "alerts.geojson").is_file():
        pytest.skip("sample run not generated; run tests/experience/fixtures/make_fixtures.py")
    run = load_run(SAMPLE_RUN_DIR, is_sample=True)
    stats = summarize(run.alerts)
    assert stats["n_alerts"] == 12
    assert stats["by_status_count"] == {"legal": 4, "disputed": 4, "illegal": 4}
    assert stats["total_ha"] == pytest.approx(run.report["total_new_disturbance_ha"], abs=0.01)
    assert run.warnings == []
