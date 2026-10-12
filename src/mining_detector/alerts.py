"""Turn a raster change mask into georeferenced alert polygons.

Output follows the contract in docs/CONTRACT.md:
    alerts.geojson is a FeatureCollection in EPSG:4326 whose features have
    id, legal_status, confidence, area_ha and detected_on properties.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine
from scipy import ndimage
from shapely.geometry import shape


def mask_to_polygons(
    mask: np.ndarray,
    transform: Affine,
    crs: str,
    prob: np.ndarray | None = None,
) -> gpd.GeoDataFrame:
    """Vectorize a boolean mask into one polygon per connected patch.

    Args:
        mask: (H, W) boolean array of alert pixels.
        transform: affine transform of the raster (pixel to map coordinates).
        crs: CRS of the raster, usually a UTM zone such as "EPSG:32719".
        prob: optional (H, W) probability map; the mean over each patch is stored as confidence.

    Returns:
        GeoDataFrame with columns id, area_ha, confidence and geometry, in `crs`.
        It is empty if the mask has no pixels.
    """
    cols = ["id", "area_ha", "confidence", "geometry"]
    if not mask.any():
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)

    labels, n = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    px_w = abs(transform.a)
    px_h = abs(transform.e)
    px_area_m2 = px_w * px_h

    records = []
    for geom_dict, value in rasterio.features.shapes(
        labels.astype(np.int32), mask=mask, transform=transform
    ):
        lid = int(value)
        if lid == 0:
            continue
        patch = labels == lid
        area_ha = float(patch.sum() * px_area_m2 / 10_000.0)
        conf = float(prob[patch].mean()) if prob is not None else 1.0
        records.append(
            {"area_ha": area_ha, "confidence": float(np.clip(conf, 0.0, 1.0)),
             "geometry": shape(geom_dict)}
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
    gdf.insert(0, "id", np.arange(1, len(gdf) + 1))
    return gdf


def build_feature_collection(
    polygons_utm: gpd.GeoDataFrame,
    legal_status: list[str],
    detected_on: str,
) -> dict:
    """Convert alert polygons to a GeoJSON FeatureCollection in EPSG:4326.

    Args:
        polygons_utm: output of mask_to_polygons (metric CRS).
        legal_status: one of "legal", "disputed", "illegal" per polygon.
        detected_on: date string "YYYY-MM-DD" of the t1 image.

    Returns:
        A dict ready for json.dump. It has no features when there are no polygons.
    """
    if len(legal_status) != len(polygons_utm):
        raise ValueError("legal_status must have one entry per polygon.")
    date.fromisoformat(detected_on)  # raises ValueError if malformed

    if polygons_utm.empty:
        return {"type": "FeatureCollection", "features": []}

    wgs = polygons_utm.to_crs("EPSG:4326")
    features = []
    for row, status in zip(wgs.itertuples(index=False), legal_status):
        features.append(
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    "id": int(row.id),
                    "legal_status": status,
                    "confidence": round(float(row.confidence), 4),
                    "area_ha": round(float(row.area_ha), 3),
                    "detected_on": detected_on,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "alerts",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
        "features": features,
    }


def write_geojson(fc: dict, path: str | Path) -> Path:
    """Write a FeatureCollection to disk as UTF-8 JSON and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    return path
