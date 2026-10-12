"""Read a finished run from disk and turn it into tables and summaries.

This module has no Streamlit code, so it can be tested with plain pytest.

A "run folder" is a folder that holds the files the engine writes:
alerts.geojson, report.json, change_map.png and probability_t1.png.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

STATUSES = ("legal", "disputed", "illegal")
REQUIRED_FILES = ("alerts.geojson", "report.json")
ALERT_COLUMNS = ["id", "legal_status", "confidence", "area_ha", "detected_on"]


@dataclass
class RunData:
    """Everything the app needs from one run folder."""

    folder: Path
    alerts: pd.DataFrame
    report: dict
    geojson: dict
    is_sample: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def change_map_path(self) -> Path:
        return self.folder / "change_map.png"

    @property
    def probability_path(self) -> Path:
        return self.folder / "probability_t1.png"


def is_run_folder(folder: Path) -> bool:
    """Return True if the folder holds every required run file."""
    return all((folder / name).is_file() for name in REQUIRED_FILES)


def find_runs(outputs_dir: Path) -> list[Path]:
    """List run folders inside outputs_dir, newest first.

    Newest is decided by the modification time of alerts.geojson, so the most
    recent run appears at the top of the sidebar selector.
    """
    if not outputs_dir.is_dir():
        return []
    runs = [p for p in outputs_dir.iterdir() if p.is_dir() and is_run_folder(p)]
    return sorted(runs, key=lambda p: (p / "alerts.geojson").stat().st_mtime, reverse=True)


def choose_data_folder(outputs_dir: Path, fixture_dir: Path) -> tuple[Path | None, bool]:
    """Pick which folder the app should show.

    Returns (folder, is_sample). Real output in outputs_dir wins. If there is
    none, the bundled sample run is used and is_sample is True so the app can
    show a warning banner.
    """
    runs = find_runs(outputs_dir)
    if runs:
        return runs[0], False
    if is_run_folder(fixture_dir):
        return fixture_dir, True
    return None, False


def load_alerts_table(geojson: dict) -> pd.DataFrame:
    """Turn a GeoJSON FeatureCollection into one row per alert.

    Only the properties listed in ALERT_COLUMNS are kept. Rows with an unknown
    legal_status raise an error, because that would silently skew the totals.
    """
    rows = []
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        rows.append({col: props.get(col) for col in ALERT_COLUMNS})
    df = pd.DataFrame(rows, columns=ALERT_COLUMNS)
    if not df.empty:
        bad = set(df["legal_status"]) - set(STATUSES)
        if bad:
            raise ValueError(f"Unknown legal_status values in alerts: {sorted(bad)}")
        df["id"] = df["id"].astype(int)
        df["confidence"] = df["confidence"].astype(float)
        df["area_ha"] = df["area_ha"].astype(float)
    return df


def load_run(folder: Path, is_sample: bool = False) -> RunData:
    """Read alerts.geojson and report.json from a run folder.

    Raises FileNotFoundError with a plain message if a required file is missing.
    """
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"This folder is missing {', '.join(missing)}. "
            "Run the demo again, or pick another run in the sidebar."
        )
    geojson = json.loads((folder / "alerts.geojson").read_text(encoding="utf-8"))
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    alerts = load_alerts_table(geojson)
    warnings = []
    if report.get("n_alerts") != len(alerts):
        warnings.append(
            f"report.json says {report.get('n_alerts')} alerts but alerts.geojson has {len(alerts)}."
        )
    return RunData(folder=folder, alerts=alerts, report=report, geojson=geojson,
                   is_sample=is_sample, warnings=warnings)


def summarize(alerts: pd.DataFrame) -> dict:
    """Count alerts and sum hectares per status. Every status is always present."""
    result = {
        "n_alerts": int(len(alerts)),
        "total_ha": round(float(alerts["area_ha"].sum()), 4) if len(alerts) else 0.0,
        "by_status_ha": {},
        "by_status_count": {},
    }
    for status in STATUSES:
        subset = alerts[alerts["legal_status"] == status]
        result["by_status_ha"][status] = round(float(subset["area_ha"].sum()), 4)
        result["by_status_count"][status] = int(len(subset))
    return result


def filter_alerts(alerts: pd.DataFrame, statuses: list[str], min_confidence: float) -> pd.DataFrame:
    """Keep alerts whose status is selected and whose confidence is at least min_confidence."""
    mask = alerts["legal_status"].isin(statuses) & (alerts["confidence"] >= min_confidence)
    return alerts[mask].reset_index(drop=True)


def alerts_to_csv(alerts: pd.DataFrame) -> bytes:
    """Return the alert table as UTF-8 CSV bytes, ready for a download button."""
    return alerts.to_csv(index=False).encode("utf-8")


def caveat_lines(report: dict) -> list[str]:
    """Return the caveats from report.json. Never filtered, so they can always be shown."""
    return [str(c) for c in report.get("caveats", [])]
