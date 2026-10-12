"""Build the interactive map for the Results page.

Uses folium. Polygons are coloured by legal status and each one shows a popup
with confidence and area when clicked.
"""

from __future__ import annotations

import folium
import pandas as pd

from lib.loaders import STATUSES

STATUS_COLOURS = {"legal": "green", "disputed": "orange", "illegal": "red"}
HEX_COLOURS = {"legal": "#2e7d32", "disputed": "#ef8c00", "illegal": "#c62828"}


def build_map(geojson: dict, alerts: pd.DataFrame, bbox: list[float]) -> folium.Map:
    """Return a folium map with a basemap toggle and one coloured polygon per alert.

    bbox is [west, south, east, north]. The map is centred on it.
    The features are taken from geojson; the table supplies the status for each id.
    """
    west, south, east, north = bbox
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=12,
                      tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery (satellite)",
        name="Satellite",
    ).add_to(fmap)
    folium.Rectangle(bounds=[[south, west], [north, east]], color="#1565c0",
                     weight=2, fill=False, tooltip="Area of interest").add_to(fmap)

    status_by_id = dict(zip(alerts["id"], alerts["legal_status"]))
    for feat in geojson.get("features", []):
        props = feat["properties"]
        status = status_by_id.get(props["id"], props["legal_status"])
        popup = (
            f"<b>Alert {props['id']}</b><br>"
            f"Status: {status}<br>"
            f"Confidence: {props['confidence']:.0%}<br>"
            f"Area: {props['area_ha']:.2f} ha<br>"
            f"Detected on: {props['detected_on']}"
        )
        folium.GeoJson(
            feat,
            style_function=lambda _f, s=status: {
                "fillColor": HEX_COLOURS[s], "color": HEX_COLOURS[s],
                "weight": 2, "fillOpacity": 0.45,
            },
            popup=folium.Popup(popup, max_width=260),
            tooltip=f"Alert {props['id']}: {status}",
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    return fmap


def legend_html() -> str:
    """A small colour key to show under the map."""
    items = "".join(
        f"<span style='display:inline-block;margin-right:16px'>"
        f"<span style='display:inline-block;width:12px;height:12px;background:{HEX_COLOURS[s]};"
        f"margin-right:6px'></span>{s.capitalize()}</span>"
        for s in STATUSES
    )
    return f"<div>{items}</div>"
