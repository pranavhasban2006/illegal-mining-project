"""Tests for the HTML report, the map builder, and the Results page rendering.

The page test runs the real Results page through Streamlit's test runner, so a
crash in the page code fails the test.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))

from lib.loaders import load_run  # noqa: E402
from lib.paths import SAMPLE_RUN_DIR  # noqa: E402
from lib.report_html import build_report_html  # noqa: E402

SAMPLE_READY = (SAMPLE_RUN_DIR / "alerts.geojson").is_file()


def _run_for_html(tmp_path):
    """A tiny run whose totals are easy to check by eye: one illegal alert of 3 ha."""
    feat = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
            "properties": {"id": 7, "legal_status": "illegal", "confidence": 0.8,
                           "area_ha": 3.0, "detected_on": "2026-06-01"}}
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "alerts.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feat]}))
    (folder / "report.json").write_text(json.dumps({
        "aoi_bbox": [0, 0, 1, 1], "t0": "2025-06-01", "t1": "2026-06-01", "n_alerts": 1,
        "caveats": ["CAVEAT-MARKER"], "data_sources": [],
    }))
    return load_run(folder)


def test_html_report_has_totals_and_caveats(tmp_path):
    run = _run_for_html(tmp_path)
    page = build_report_html(run.report, run.alerts, is_sample=False)
    assert "1 alerts" in page
    assert "3.00 ha of new disturbance" in page
    assert "CAVEAT-MARKER" in page
    assert "SAMPLE DATA" not in page


def test_html_report_marks_sample_data(tmp_path):
    run = _run_for_html(tmp_path)
    page = build_report_html(run.report, run.alerts, is_sample=True)
    assert "SAMPLE DATA" in page


def test_html_report_escapes_text(tmp_path):
    run = _run_for_html(tmp_path)
    run.report["caveats"] = ["<script>bad()</script>"]
    page = build_report_html(run.report, run.alerts, is_sample=False)
    assert "<script>bad()</script>" not in page
    assert "&lt;script&gt;" in page


def test_map_has_one_polygon_per_alert_and_satellite_layer(tmp_path):
    from lib.maps import build_map

    run = _run_for_html(tmp_path)
    fmap = build_map(run.geojson, run.alerts, [0, 0, 1, 1])
    rendered = fmap._repr_html_()
    assert "Satellite" in rendered
    assert "Alert 7" in rendered
    assert "#c62828" in rendered  # red for illegal


@pytest.mark.skipif(not SAMPLE_READY, reason="sample run not generated")
def test_results_page_renders_sample_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "app" / "pages" / "1_Results.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    banners = [w.value for w in at.warning]
    assert any("SAMPLE" in b for b in banners)
    assert len(at.metric) >= 4


def test_datasets_page_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "app" / "pages" / "Datasets.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Checklist" in h.value for h in at.subheader)


def test_models_page_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO / "app" / "pages" / "Models.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Honesty panel" in w.value for w in at.warning)
