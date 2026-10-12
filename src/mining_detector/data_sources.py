"""Find Sentinel-2 scenes for an area and date.

Primary catalogue: Element 84 Earth Search (AWS open data, no account needed).
Fallback: Microsoft Planetary Computer, used automatically if Earth Search fails.
Both serve the same Sentinel-2 L2A (atmospherically corrected) products.

For each request we pick the least cloudy scene within +/- `window_days` of the
requested date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

log = logging.getLogger(__name__)

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
PLANETARY_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


@dataclass
class SceneChoice:
    """A chosen Sentinel-2 scene. item is the raw STAC item (already signed if needed)."""

    item: Any
    source: str
    item_id: str
    acquired: date
    cloud_pct: float
    epsg: int | None
    baseline: str
    boa_offset_applied: bool = False  # True if the catalogue already removed the +1000 DN offset

    def describe(self) -> str:
        return (f"{self.item_id} acquired {self.acquired.isoformat()} "
                f"({self.cloud_pct:.1f}% cloud, source: {self.source})")


def _search(url: str, bbox, start: date, end: date, max_cloud: float):
    """Run one STAC search and return a list of items (raises on network or API errors)."""
    from pystac_client import Client

    client = Client.open(url)
    search = client.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=200,
    )
    return list(search.items())


def _sign_if_needed(items: list, source: str) -> list:
    """Planetary Computer serves private-looking URLs that must be signed with a SAS token."""
    if source != "planetary_computer":
        return items
    import planetary_computer

    return [planetary_computer.sign(i) for i in items]


def _best(items: list, target: date) -> Any | None:
    """Least cloudy item, with ties broken by closeness to the target date."""
    if not items:
        return None
    return min(
        items,
        key=lambda it: (
            float(it.properties.get("eo:cloud_cover", 100.0)),
            abs((it.datetime.date() - target).days),
        ),
    )


def _to_choice(item, source: str, target: date) -> SceneChoice:
    props = item.properties
    return SceneChoice(
        item=item,
        source=source,
        item_id=item.id,
        acquired=item.datetime.date(),
        cloud_pct=float(props.get("eo:cloud_cover", float("nan"))),
        epsg=props.get("proj:epsg"),
        baseline=str(props.get("s2:processing_baseline", "")),
        boa_offset_applied=bool(props.get("earthsearch:boa_offset_applied", False)),
    )


def search_scenes(
    bbox: tuple[float, float, float, float],
    date_str: str,
    max_cloud: float = 20.0,
    window_days: int = 15,
) -> SceneChoice:
    """Find the least cloudy Sentinel-2 scene near a date.

    Args:
        bbox: (west, south, east, north) in degrees.
        date_str: target date, "YYYY-MM-DD".
        max_cloud: maximum scene cloud cover in percent.
        window_days: search this many days either side of the target date.

    Returns:
        SceneChoice for the chosen scene.

    Raises:
        RuntimeError: if no scene passes the cloud limit, or every catalogue fails.
            The message says what to change.
    """
    target = date.fromisoformat(date_str)
    start = target - timedelta(days=window_days)
    end = target + timedelta(days=window_days)

    errors: list[str] = []
    for source, url in (("earth_search", EARTH_SEARCH_URL), ("planetary_computer", PLANETARY_URL)):
        try:
            log.info("Searching %s for Sentinel-2 scenes %s to %s", source, start, end)
            items = _search(url, bbox, start, end, max_cloud)
        except Exception as exc:  # network, DNS, HTTP or parsing errors
            msg = f"{source} search failed: {type(exc).__name__}: {exc}"
            log.warning(msg)
            errors.append(msg)
            continue

        items = _sign_if_needed(items, source)
        best = _best(items, target)
        if best is None:
            raise RuntimeError(
                f"No scene under {max_cloud:g}% cloud within {window_days} days of {date_str}. "
                f"Try a larger --max-cloud (for example 40) or a wider date window."
            )
        return _to_choice(best, source, target)

    raise RuntimeError(
        "Could not reach either satellite catalogue (Earth Search or Planetary Computer). "
        "Check your internet connection and try again. Details: " + "; ".join(errors)
    )
