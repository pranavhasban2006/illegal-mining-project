"""Helpers for the Analyze wizard: area sizes, coordinate parsing, presets and dates.

No Streamlit here, so these are testable with plain pytest.
"""

from __future__ import annotations

import math
import re
from datetime import date, timedelta

# Example areas. The first is the engine's own default example (docs/ENGINE_NOTES.md).
# The second is an approximate box around the Ankobra basin in Ghana. It is a starting
# point for exploring, not a surveyed boundary, so adjust it before drawing conclusions.
PRESETS = {
    "Example: Peru (engine default)": [-72.10, -14.45, -71.95, -14.30],
    "Example: Ghana galamsey (approximate)": [-2.45, 5.70, -2.00, 6.10],
}

DEFAULT_MAX_KM2 = 400.0  # the engine's default limit (max_aoi_km2 in the contract)


def bbox_area_km2(bbox: list[float]) -> float:
    """Approximate area of a lon/lat box in km^2, using the box's mid-latitude."""
    w, s, e, n = bbox
    mid_lat = math.radians((s + n) / 2)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(mid_lat)
    return abs(e - w) * km_per_deg_lon * abs(n - s) * km_per_deg_lat


def validate_bbox(bbox: list[float], max_km2: float = DEFAULT_MAX_KM2) -> list[str]:
    """Return a list of plain-English problems with the box. An empty list means it is usable."""
    problems = []
    if len(bbox) != 4:
        return ["The area needs four numbers: west, south, east, north."]
    w, s, e, n = bbox
    if not (-180 <= w <= 180 and -180 <= e <= 180):
        problems.append("Longitude (west and east) must be between -180 and 180.")
    if not (-90 <= s <= 90 and -90 <= n <= 90):
        problems.append("Latitude (south and north) must be between -90 and 90.")
    if w >= e:
        problems.append("West must be less than east.")
    if s >= n:
        problems.append("South must be less than north.")
    if not problems:
        area = bbox_area_km2(bbox)
        if area > max_km2:
            problems.append(
                f"This area is {area:,.0f} km². The limit is {max_km2:,.0f} km². "
                "Draw a smaller box so the satellite download stays manageable."
            )
    return problems


_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_bbox_text(text: str) -> list[float]:
    """Read four numbers from text such as '-72.1, -14.45, -71.95, -14.3' or space separated.

    Raises ValueError with a plain message if there are not exactly four numbers.
    """
    found = _NUMBER.findall(text or "")
    if len(found) != 4:
        raise ValueError(
            "Enter four numbers in this order: west, south, east, north "
            "(for example: -72.10 -14.45 -71.95 -14.30)."
        )
    return [float(x) for x in found]


def default_dates(t1: date) -> tuple[date, date]:
    """Return (t0, t1) with t0 one year before t1.

    Comparing the same calendar date a year apart keeps the season the same, so changes
    in the vegetation from season to season do not look like mining.
    """
    try:
        t0 = t1.replace(year=t1.year - 1)
    except ValueError:  # 29 February has no match in a non-leap year
        t0 = t1 - timedelta(days=365)
    return t0, t1


def format_dates_note(t0: date, t1: date) -> str:
    """A short plain-English reason for the chosen dates."""
    return (f"Compares {t0.isoformat()} with {t1.isoformat()}. Both dates are in the same season, "
            "so seasonal greenery changes are less likely to be mistaken for mining.")
