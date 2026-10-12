"""Tests for the Analyze wizard helpers (app/lib/area.py and app/lib/runner.py).

The runner tests use a small stand-in function to check threading and error handling.
They say nothing about detection quality.
"""

import sys
import time
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "app"))

from lib.area import (  # noqa: E402
    PRESETS, bbox_area_km2, default_dates, parse_bbox_text, validate_bbox,
)
from lib.runner import RunJob, what_to_try  # noqa: E402


def test_area_of_one_degree_box_near_equator_matches_formula():
    # The box centre is at latitude 0.5, so longitude degrees are scaled by cos(0.5 deg).
    import math

    area = bbox_area_km2([0.0, 0.0, 1.0, 1.0])
    assert area == pytest.approx(110.574 * 111.320 * math.cos(math.radians(0.5)), rel=1e-9)


def test_area_shrinks_with_latitude():
    assert bbox_area_km2([0, 60, 1, 61]) < bbox_area_km2([0, 0, 1, 1])


def test_valid_small_box_has_no_problems():
    assert validate_bbox(PRESETS["Example: Peru (engine default)"]) == []


def test_box_over_limit_gives_plain_message():
    problems = validate_bbox([0.0, 0.0, 5.0, 5.0], max_km2=100)
    assert problems and "limit is 100 km²" in problems[0]


def test_inverted_box_is_rejected():
    problems = validate_bbox([1.0, 0.0, 0.0, 1.0])
    assert any("West must be less than east" in p for p in problems)


def test_out_of_range_coordinates_are_rejected():
    problems = validate_bbox([-200.0, 0.0, -199.0, 1.0])
    assert any("Longitude" in p for p in problems)


def test_parse_accepts_commas_and_spaces():
    assert parse_bbox_text("-72.10, -14.45, -71.95, -14.30") == [-72.10, -14.45, -71.95, -14.30]
    assert parse_bbox_text("-72.10 -14.45 -71.95 -14.30") == [-72.10, -14.45, -71.95, -14.30]


def test_parse_rejects_wrong_count():
    with pytest.raises(ValueError, match="four numbers"):
        parse_bbox_text("1 2 3")


def test_default_dates_are_one_year_apart_on_same_day():
    assert default_dates(date(2025, 7, 27)) == (date(2024, 7, 27), date(2025, 7, 27))


def test_default_dates_handle_leap_day():
    # 29 February 2024 has no counterpart in 2023, so the code steps back 365 days instead.
    t0, t1 = default_dates(date(2024, 2, 29))
    assert t1 == date(2024, 2, 29)
    assert t0 == date(2023, 3, 1)


def test_run_job_reports_progress_and_result():
    job = RunJob()

    def fake(progress, value):
        progress(0.5, "halfway")
        return value * 2

    job.start(fake, value=21)
    deadline = time.time() + 5
    while job.is_running() and time.time() < deadline:
        time.sleep(0.01)
    assert job.done and job.error is None
    assert job.result == 42
    assert job.fraction == 1.0
    assert "halfway" in job.log


def test_run_job_captures_error_message():
    job = RunJob()

    def fails(progress):
        raise ValueError("The area is too big. Draw a smaller box.")

    job.start(fails)
    deadline = time.time() + 5
    while job.is_running() and time.time() < deadline:
        time.sleep(0.01)
    assert job.error == "The area is too big. Draw a smaller box."
    assert job.error_type == "ValueError"


def test_progress_fraction_is_clamped():
    job = RunJob()
    job.progress(1.7, "over")
    assert job.fraction == 1.0
    job.progress(-0.3, "under")
    assert job.fraction == 0.0


def test_what_to_try_matches_common_failures():
    assert "smaller box" in what_to_try("ValueError", "This area is too large")
    assert "different dates" in what_to_try("RuntimeError", "No pixels are both labelled and clear")
    assert "internet" in what_to_try("RuntimeError", "catalogue download timeout")
