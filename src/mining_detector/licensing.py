"""Compare detected disturbance with concession polygons to label each alert.

Labels:
    legal    - the disturbance lies inside an active concession.
    disputed - it touches an inactive or expired concession, or sits within the
               buffer distance of any concession (boundary uncertainty or stale licence).
    illegal  - it is outside every concession and buffer.

Everything here works in a metric projection (metres), so buffers are real distances.
"""

from __future__ import annotations

import logging

import geopandas as gpd

log = logging.getLogger(__name__)

ACTIVE = "active"
INACTIVE_STATUSES = {"inactive", "expired", "cancelled", "canceled", "lapsed"}


def classify_alerts(
    alerts: gpd.GeoDataFrame,
    concessions: gpd.GeoDataFrame | None,
    buffer_m: float = 500.0,
) -> tuple[list[str], list[str]]:
    """Return a legal status for each alert polygon, in the same order as the input.

    Args:
        alerts: polygons of new disturbance, in a metric CRS (for example UTM).
        concessions: concession polygons with an optional "status" column. Rows with
            no status are treated as active, and a note is returned about that.
        buffer_m: distance in metres around any concession that counts as disputed.

    Returns:
        (labels, notes): labels is a list of "legal", "disputed" or "illegal";
        notes is a list of plain-English remarks about the inputs.

    Raises:
        ValueError: if the two layers are not in the same metric CRS.
    """
    notes: list[str] = []
    if alerts.empty:
        return [], notes
    if concessions is None or concessions.empty:
        notes.append("No concession polygons were supplied, so every alert is labelled illegal.")
        return ["illegal"] * len(alerts), notes

    if concessions.crs != alerts.crs:
        raise ValueError(
            f"Alert CRS ({alerts.crs}) and concession CRS ({concessions.crs}) differ. "
            "Reproject the concessions to the alert CRS first."
        )

    conc = concessions.copy()
    if "status" not in conc.columns:
        notes.append(
            "Concession file has no 'status' column; all concessions were treated as active."
        )
        conc["status"] = ACTIVE
    conc["status"] = conc["status"].fillna(ACTIVE).astype(str).str.strip().str.lower()

    is_inactive = conc["status"].isin(INACTIVE_STATUSES)
    active_geom = conc.loc[~is_inactive].geometry
    inactive_geom = conc.loc[is_inactive].geometry

    active_union = active_geom.union_all() if len(active_geom) else None
    inactive_union = inactive_geom.union_all() if len(inactive_geom) else None
    any_union = conc.geometry.union_all()
    buffered_union = any_union.buffer(buffer_m)

    labels: list[str] = []
    for geom in alerts.geometry:
        if active_union is not None and geom.intersects(active_union):
            labels.append("legal")
        elif inactive_union is not None and geom.intersects(inactive_union):
            labels.append("disputed")
        elif geom.intersects(buffered_union):
            labels.append("disputed")
        else:
            labels.append("illegal")

    log.info(
        "Licence check: %d legal, %d disputed, %d illegal",
        labels.count("legal"), labels.count("disputed"), labels.count("illegal"),
    )
    return labels, notes
