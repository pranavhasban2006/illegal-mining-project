"""O4: Cross-reference detected change against licensed-mining-lease boundaries.

Produces a per-pixel "suspicious" mask (new disturbance NOT inside any
active licensed lease polygon) plus a confidence score blending classifier
probability with magnitude of temporal change.

Also supports a finer 3-tier legal-status classification
(classify_legal_status): a concession's administrative status in Peru
isn't binary. New disturbance inside an ACTIVE/titled concession is legal;
inside a concession that is frozen ("Bloqueado"), pending ("Tramite") or
expired ("Extinguido") is DISPUTED -- a different regulatory action than
outright illegal mining; and disturbance with no concession at all is
ILLEGAL. Collapsing DISPUTED and ILLEGAL into one "suspicious" bucket (the
binary path below) loses information a regulator would want when deciding
where to send an inspection team first.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from affine import Affine


def rasterize_leases(
    lease_gdf: gpd.GeoDataFrame,
    out_shape: tuple[int, int],
    transform: Affine,
    target_crs: str,
) -> np.ndarray:
    """Burns lease polygons onto the scene grid. Returns (H,W) bool: True=licensed."""
    if lease_gdf is None or len(lease_gdf) == 0:
        return np.zeros(out_shape, dtype=bool)

    gdf = lease_gdf.to_crs(target_crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        return np.zeros(out_shape, dtype=bool)

    burned = rasterize(
        shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8
    )
    return burned.astype(bool)


def suspicious_area_mask(
    new_disturbance_mask: np.ndarray,
    licensed_mask: np.ndarray,
) -> np.ndarray:
    """New disturbance that falls outside every known licensed lease."""
    return (new_disturbance_mask.astype(bool) & ~licensed_mask).astype(np.uint8)


def suspicious_confidence(
    suspicious_mask: np.ndarray,
    detection_proba: np.ndarray,
    change_conf: np.ndarray,
) -> np.ndarray:
    """Confidence score in [0,1] per pixel: only meaningful where suspicious_mask==1.

    Blends classifier confidence and magnitude of temporal change -- a pixel
    that is both strongly classified as mining AND shows a large jump in
    probability since t0 scores highest.
    """
    score = 0.6 * detection_proba + 0.4 * change_conf
    return np.where(suspicious_mask.astype(bool), score, 0.0)


class LegalStatus(IntEnum):
    NOT_MINING = 0
    LEGAL = 1       # new disturbance inside an active/titled concession
    DISPUTED = 2    # inside a concession that is frozen/pending/expired
    ILLEGAL = 3     # no concession at all


LEGAL_STATUS_NAMES = {
    LegalStatus.NOT_MINING: "not_mining",
    LegalStatus.LEGAL: "legal",
    LegalStatus.DISPUTED: "disputed",
    LegalStatus.ILLEGAL: "illegal",
}


def classify_legal_status(
    new_disturbance_mask: np.ndarray,
    active_mask: np.ndarray,
    any_concession_mask: np.ndarray,
) -> np.ndarray:
    """Returns (H,W) uint8 of LegalStatus values for new-disturbance pixels.

    active_mask: rasterized active/titled concessions (real_labels.active_concessions).
    any_concession_mask: rasterized concessions of ANY status -- a superset of active_mask.
    """
    status = np.full(new_disturbance_mask.shape, LegalStatus.NOT_MINING, dtype=np.uint8)
    new = new_disturbance_mask.astype(bool)

    status[new & active_mask] = LegalStatus.LEGAL
    status[new & any_concession_mask & ~active_mask] = LegalStatus.DISPUTED
    status[new & ~any_concession_mask] = LegalStatus.ILLEGAL
    return status


def status_confidence(
    legal_status: np.ndarray,
    detection_proba: np.ndarray,
    change_conf: np.ndarray,
) -> np.ndarray:
    """Like suspicious_confidence, but scored for DISPUTED and ILLEGAL only.

    DISPUTED pixels get a mild confidence discount: a frozen or pending
    concession may yet be resolved administratively, so it's a lower
    enforcement priority than land with no concession claim at all --
    without hiding it from the alert output entirely.
    """
    base = 0.6 * detection_proba + 0.4 * change_conf
    score = np.zeros_like(base)
    score[legal_status == LegalStatus.ILLEGAL] = base[legal_status == LegalStatus.ILLEGAL]
    disputed = legal_status == LegalStatus.DISPUTED
    score[disputed] = 0.8 * base[disputed]
    return score
