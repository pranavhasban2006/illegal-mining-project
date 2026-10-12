"""Write a small, clearly labelled SAMPLE run so the app can be built and tested
before the real pipeline produces output.

Run from the repo root:
    python tests/experience/fixtures/make_fixtures.py

It writes four files into tests/experience/fixtures/sample_run/:
    alerts.geojson, report.json, change_map.png, probability_t1.png

Everything here is synthetic. The 12 sites are invented rectangles placed near
La Pampa, Peru (lon -70.9, lat -12.9). Their areas are computed from the
rectangle dimensions in metres, so tests can check the exact answer. These
numbers say nothing about real mining.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files only, no windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sample_run"

AOI_BBOX = [-71.0, -13.0, -70.8, -12.8]  # west, south, east, north
T0 = "2025-06-01"
T1 = "2026-06-01"
CENTER_LAT = -12.9

# Each site: (centre lon, centre lat, width m, height m, legal_status, confidence)
# Twelve sites: 4 legal, 4 disputed, 4 illegal.
SITES = [
    (-70.94, -12.93, 180.0, 160.0, "legal", 0.91),
    (-70.91, -12.96, 220.0, 140.0, "legal", 0.88),
    (-70.87, -12.90, 150.0, 150.0, "legal", 0.79),
    (-70.84, -12.94, 200.0, 120.0, "legal", 0.84),
    (-70.96, -12.88, 260.0, 210.0, "disputed", 0.66),
    (-70.92, -12.85, 190.0, 170.0, "disputed", 0.58),
    (-70.89, -12.99, 300.0, 180.0, "disputed", 0.71),
    (-70.82, -12.87, 210.0, 200.0, "disputed", 0.62),
    (-70.95, -12.97, 340.0, 260.0, "illegal", 0.94),
    (-70.90, -12.92, 280.0, 230.0, "illegal", 0.97),
    (-70.86, -12.82, 240.0, 190.0, "illegal", 0.89),
    (-70.81, -12.96, 310.0, 250.0, "illegal", 0.93),
]

M_PER_DEG_LAT = 110_574.0


def m_per_deg_lon(lat: float) -> float:
    """Metres per degree of longitude at a given latitude."""
    return 111_320.0 * math.cos(math.radians(lat))


def rectangle_ring(lon: float, lat: float, w_m: float, h_m: float) -> list[list[float]]:
    """Return a closed rectangle ring (lon, lat) of size w_m x h_m metres."""
    dlon = (w_m / 2.0) / m_per_deg_lon(lat)
    dlat = (h_m / 2.0) / M_PER_DEG_LAT
    corners = [
        (lon - dlon, lat - dlat),
        (lon + dlon, lat - dlat),
        (lon + dlon, lat + dlat),
        (lon - dlon, lat + dlat),
    ]
    ring = [[round(x, 6), round(y, 6)] for x, y in corners]
    ring.append(ring[0])  # GeoJSON rings must close
    return ring


def build_alerts() -> dict:
    """Build the alerts FeatureCollection. Area comes from the metre dimensions."""
    features = []
    for i, (lon, lat, w, h, status, conf) in enumerate(SITES, start=1):
        area_ha = round(w * h / 10_000.0, 4)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [rectangle_ring(lon, lat, w, h)]},
                "properties": {
                    "id": i,
                    "legal_status": status,
                    "confidence": conf,
                    "area_ha": area_ha,
                    "detected_on": T1,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "SAMPLE alerts (synthetic)",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }


def build_report(alerts: dict) -> dict:
    """Build report.json from the alerts, so totals always match the features."""
    by_status = {"legal": 0.0, "disputed": 0.0, "illegal": 0.0}
    for feat in alerts["features"]:
        props = feat["properties"]
        by_status[props["legal_status"]] += props["area_ha"]
    by_status = {k: round(v, 4) for k, v in by_status.items()}
    total = round(sum(by_status.values()), 4)

    return {
        "aoi_bbox": AOI_BBOX,
        "t0": T0,
        "t1": T1,
        "n_alerts": len(alerts["features"]),
        "total_new_disturbance_ha": total,
        "by_legal_status": by_status,
        "model": {
            "best": "random_forest",
            "candidates": [
                {"name": "logistic_regression", "f1_train": 0.78, "f1_test": 0.76, "overfit_flag": False},
                {"name": "gradient_boosting", "f1_train": 0.99, "f1_test": 0.71, "overfit_flag": True},
                {"name": "random_forest", "f1_train": 0.92, "f1_test": 0.84, "overfit_flag": False},
            ],
            "feature_importance": {
                "ndvi_drop": 0.41,
                "bare_soil_index": 0.27,
                "ndwi_change": 0.14,
                "texture_contrast": 0.11,
                "slope": 0.07,
            },
        },
        "data_sources": [
            "SAMPLE fixture: synthetic values, not real Sentinel-2 imagery",
        ],
        "caveats": [
            "SAMPLE DATA: every alert in this folder is invented for testing the app.",
            "No real satellite image was analysed to produce these numbers.",
            "F1 scores are on a synthetic split and must not be quoted as real accuracy.",
        ],
    }


def draw_change_map(alerts: dict, path: Path) -> None:
    """Draw a quicklook image of the alert polygons on a flat background."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.set_facecolor("#f4efe3")
    colours = {"legal": "#2e7d32", "disputed": "#ef8c00", "illegal": "#c62828"}
    for feat in alerts["features"]:
        ring = np.array(feat["geometry"]["coordinates"][0])
        ax.fill(ring[:, 0], ring[:, 1], color=colours[feat["properties"]["legal_status"]], alpha=0.85)
    ax.set_xlim(AOI_BBOX[0], AOI_BBOX[2])
    ax.set_ylim(AOI_BBOX[1], AOI_BBOX[3])
    ax.set_title("SAMPLE change map (synthetic)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_probability(alerts: dict, path: Path) -> None:
    """Draw a synthetic probability-of-disturbance image that peaks at each alert."""
    xs = np.linspace(AOI_BBOX[0], AOI_BBOX[2], 240)
    ys = np.linspace(AOI_BBOX[1], AOI_BBOX[3], 160)
    grid_x, grid_y = np.meshgrid(xs, ys)
    prob = np.zeros_like(grid_x)
    for feat in alerts["features"]:
        ring = np.array(feat["geometry"]["coordinates"][0])
        cx, cy = ring[:-1, 0].mean(), ring[:-1, 1].mean()
        prob = np.maximum(prob, feat["properties"]["confidence"] * np.exp(
            -(((grid_x - cx) / 0.012) ** 2 + ((grid_y - cy) / 0.010) ** 2)
        ))
    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    im = ax.imshow(prob, origin="lower", extent=AOI_BBOX, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="probability of new disturbance")
    ax.set_title("SAMPLE probability map (synthetic)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alerts = build_alerts()
    report = build_report(alerts)
    (OUT_DIR / "alerts.geojson").write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draw_change_map(alerts, OUT_DIR / "change_map.png")
    draw_probability(alerts, OUT_DIR / "probability_t1.png")
    (OUT_DIR / "README.txt").write_text(
        "SAMPLE DATA. Synthetic values for testing the app. Not real detections.\n",
        encoding="utf-8",
    )
    print(f"Wrote sample run to {OUT_DIR}")


if __name__ == "__main__":
    main()
