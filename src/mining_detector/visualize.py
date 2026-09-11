"""Produces a presentation-ready map of detected alerts for the demo.

Renders a static PNG (true-color t1 basemap + alert polygons colored by
legal_status) and an interactive standalone Leaflet HTML map (pan/zoom,
click-for-details) -- the HTML needs no server or internet-dependent build
step beyond the Leaflet CDN tiles, so it's the one to actually show live.
"""

from __future__ import annotations

import json
import os

import numpy as np
import geopandas as gpd
from PIL import Image
from rasterio.warp import transform_bounds

STATUS_COLORS = {
    "illegal": "#D64545",
    "disputed": "#E8A33D",
    "legal": "#3D7A57",
}


def render_basemap_png(
    bands: np.ndarray, out_path: str, stretch_percentiles=(2, 98), max_dim: int = 1600
) -> str:
    """True-color (B04,B03,B02) PNG of the AOI, percentile-stretched and
    downsampled to at most max_dim px on the long side for fast browser loading.
    """
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    lo, hi = np.nanpercentile(rgb, stretch_percentiles)
    stretched = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    img = (stretched * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    scale = max_dim / max(pil_img.size)
    if scale < 1:
        new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    pil_img.save(out_path)
    return out_path


def build_interactive_map(
    alert_gdf: gpd.GeoDataFrame,
    aoi_crs: str,
    basemap_png_path: str,
    aoi_bounds_native: tuple[float, float, float, float],
    out_path: str,
    title: str = "Illegal Mining Detector -- Alert Map",
) -> str:
    """Writes a standalone Leaflet HTML map: basemap image overlay + alert
    polygons (reprojected to WGS84) colored/popup-labeled by legal_status.
    """
    minx, miny, maxx, maxy = transform_bounds(aoi_crs, "EPSG:4326", *aoi_bounds_native)
    center_lat, center_lon = (miny + maxy) / 2, (minx + maxx) / 2

    alerts_wgs84 = alert_gdf.to_crs("EPSG:4326") if len(alert_gdf) else alert_gdf
    features_geojson = json.loads(alerts_wgs84.to_json()) if len(alert_gdf) else {"type": "FeatureCollection", "features": []}

    basemap_filename = os.path.basename(basemap_png_path)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html, body {{ margin: 0; height: 100%; font-family: system-ui, sans-serif; }}
  #map {{ height: 100%; }}
  .legend {{
    position: absolute; bottom: 24px; right: 12px; z-index: 1000;
    background: white; padding: 10px 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-size: 13px; line-height: 1.8;
  }}
  .legend h4 {{ margin: 0 0 6px; font-size: 13px; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }}
  .title-box {{
    position: absolute; top: 12px; left: 50px; z-index: 1000;
    background: white; padding: 8px 14px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-size: 14px; font-weight: 600;
  }}
</style>
</head>
<body>
<div id="map"></div>
<div class="title-box">{title}</div>
<div class="legend">
  <h4>Legal status</h4>
  <div><span class="swatch" style="background:{STATUS_COLORS['illegal']}"></span>Illegal (no concession)</div>
  <div><span class="swatch" style="background:{STATUS_COLORS['disputed']}"></span>Disputed (frozen/pending concession)</div>
  <div><span class="swatch" style="background:{STATUS_COLORS['legal']}"></span>Legal (active concession)</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  // No external tile provider -- OSM's tile servers block direct/offline
  // usage like this (their usage policy requires registered apps with a
  // proper referrer). The Sentinel-2 basemap image below IS the map; no
  // background tile layer is needed and this keeps the demo fully
  // offline-safe.
  var map = L.map('map', {{ crs: L.CRS.EPSG3857, minZoom: 10 }}).setView([{center_lat}, {center_lon}], 13);

  var imageBounds = [[{miny}, {minx}], [{maxy}, {maxx}]];
  L.imageOverlay('{basemap_filename}', imageBounds, {{opacity: 1}}).addTo(map);
  map.setMaxBounds(imageBounds);

  var statusColors = {json.dumps(STATUS_COLORS)};
  var alertData = {json.dumps(features_geojson)};

  L.geoJSON(alertData, {{
    style: function(feature) {{
      var status = feature.properties.legal_status;
      return {{
        color: statusColors[status] || '#888',
        weight: 1.5,
        fillOpacity: 0.55,
      }};
    }},
    onEachFeature: function(feature, layer) {{
      var p = feature.properties;
      layer.bindPopup(
        '<b>' + p.alert_id + '</b><br>' +
        'Status: ' + p.legal_status + '<br>' +
        'Confidence: ' + (p.confidence * 100).toFixed(1) + '%<br>' +
        'Area: ' + p.area_ha.toFixed(2) + ' ha'
      );
    }},
  }}).addTo(map);

  map.fitBounds(imageBounds);
</script>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
