"""Where the app looks for data. Kept in one place so every page agrees."""

from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = APP_DIR.parent
OUTPUTS_DIR = REPO_DIR / "outputs"
SAMPLE_RUN_DIR = REPO_DIR / "tests" / "experience" / "fixtures" / "sample_run"
