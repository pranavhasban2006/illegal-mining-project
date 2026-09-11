"""O5: Export detections as geo-referenced GeoJSON alerts.

Vectorizes the suspicious-area raster into polygons, attaches a confidence
score (mean of the per-pixel confidence within each polygon) and area, and
writes a GeoJSON FeatureCollection suitable for a downstream notification
system.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import geopandas as gpd
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape
from affine import Affine


def vectorize_alerts(
    suspicious_mask: np.ndarray,
    confidence_map: np.ndarray,
    transform: Affine,
    crs: str,
    observation_date: str,
    min_area_m2: float = 500.0,
) -> gpd.GeoDataFrame:
    """Turns the suspicious binary mask into polygon alerts with metadata."""
    mask_bool = suspicious_mask.astype(bool)
    records = []

    for geom, value in rio_shapes(
        suspicious_mask.astype(np.uint8), mask=mask_bool, transform=transform
    ):
        if value != 1:
            continue
        poly = shp_shape(geom)
        if poly.area < min_area_m2:
            continue
        records.append({"geometry": poly})

    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "confidence", "area_ha", "observation_date", "alert_id"],
            geometry="geometry",
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # Compute confidence/area per polygon by rasterizing each geometry's mask.
    from rasterio.features import geometry_mask

    confidences = []
    for geom in gdf.geometry:
        gmask = geometry_mask(
            [geom], out_shape=confidence_map.shape, transform=transform, invert=True
        )
        vals = confidence_map[gmask]
        confidences.append(float(vals.mean()) if vals.size else 0.0)

    gdf["confidence"] = confidences
    gdf["area_ha"] = gdf.geometry.area / 10_000.0
    gdf["observation_date"] = observation_date
    gdf["alert_id"] = [
        f"ALERT-{observation_date}-{i:04d}" for i in range(len(gdf))
    ]
    gdf["generated_at"] = datetime.now(timezone.utc).isoformat()

    return gdf.sort_values("confidence", ascending=False).reset_index(drop=True)


def vectorize_status_alerts(
    legal_status: np.ndarray,
    confidence_map: np.ndarray,
    transform: Affine,
    crs: str,
    observation_date: str,
    min_area_m2: float = 500.0,
) -> gpd.GeoDataFrame:
    """Like vectorize_alerts, but sourced from a 3-tier legal_status raster
    (see licensing.classify_legal_status): each polygon carries its own
    legal_status ("disputed" or "illegal" -- "legal"/"not_mining" pixels are
    never vectorized as alerts). rio_shapes groups by raster value, so
    disputed and illegal regions naturally come out as separate polygons
    even where they're spatially adjacent.
    """
    from . import licensing
    from rasterio.features import geometry_mask

    alertable = {int(licensing.LegalStatus.DISPUTED), int(licensing.LegalStatus.ILLEGAL)}
    mask_bool = np.isin(legal_status, list(alertable))
    records = []

    for geom, value in rio_shapes(legal_status.astype(np.uint8), mask=mask_bool, transform=transform):
        value = int(value)
        if value not in alertable:
            continue
        poly = shp_shape(geom)
        if poly.area < min_area_m2:
            continue
        records.append({"geometry": poly, "legal_status": licensing.LEGAL_STATUS_NAMES[value]})

    if not records:
        return gpd.GeoDataFrame(
            columns=["geometry", "legal_status", "confidence", "area_ha", "observation_date", "alert_id"],
            geometry="geometry",
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    confidences = []
    for geom in gdf.geometry:
        gmask = geometry_mask([geom], out_shape=confidence_map.shape, transform=transform, invert=True)
        vals = confidence_map[gmask]
        confidences.append(float(vals.mean()) if vals.size else 0.0)

    gdf["confidence"] = confidences
    gdf["area_ha"] = gdf.geometry.area / 10_000.0
    gdf["observation_date"] = observation_date
    gdf["alert_id"] = [f"ALERT-{observation_date}-{i:04d}" for i in range(len(gdf))]
    gdf["generated_at"] = datetime.now(timezone.utc).isoformat()

    return gdf.sort_values("confidence", ascending=False).reset_index(drop=True)


def write_geojson(gdf: gpd.GeoDataFrame, out_path: str, target_crs: str = "EPSG:4326"):
    """Reproject to WGS84 (standard GeoJSON convention) and write to disk."""
    gdf.to_crs(target_crs).to_file(out_path, driver="GeoJSON")
    return out_path
