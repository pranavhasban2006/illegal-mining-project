"""Configuration for a detection run.

Every value has a sensible default, and all of them can be overridden either by
constructing a Config in Python or by passing command-line flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Example area: a gold-mining cluster in southern Peru (near Cusco, around -72.0, -14.4).
# Chosen because the Global Mining Footprint file has 13 usable footprints here. The earlier
# La Pampa box (-69.60, -12.70, -69.45, -12.58) has none, so training labels cannot come
# from that file there.
DEFAULT_BBOX = (-72.10, -14.45, -71.95, -14.30)  # (west, south, east, north), EPSG:4326

# Sentinel-2 L2A bands used by the pipeline (10 m and 20 m bands, resampled to 10 m).
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")

# Scene-level cloud limit used when searching for Sentinel-2 scenes.
DEFAULT_MAX_CLOUD = 20.0

# Days either side of the requested date that we search for a usable scene.
SEARCH_WINDOW_DAYS = 15


@dataclass
class Config:
    """All settings for one detection run.

    Attributes:
        bbox: area of interest as (west, south, east, north) in degrees.
        t0: earlier date, "YYYY-MM-DD".
        t1: later date, "YYYY-MM-DD".
        out_dir: folder where alerts.geojson, report.json and PNGs are written.
        cache_dir: folder for cached, pre-cropped Sentinel-2 arrays.
        concessions_path: optional GeoJSON/GPKG of concession polygons (with a
            "status" column whose values are "active", "inactive" or "expired").
        resolution_m: working pixel size in metres.
        max_cloud: maximum scene cloud cover in percent.
        search_window_days: how far from the requested date a scene may be.
        min_area_ha: disturbances smaller than this are dropped as noise.
        p_high: probability above which a t1 pixel counts as mining.
        p_low: probability below which a t0 pixel counts as not yet mined.
        buffer_m: distance around an active concession that is still treated as
            legal, and the ring around an expired one that counts as "disputed".
        max_aoi_km2: AOIs larger than this are refused, to keep runtimes bounded.
        seed: random seed, so repeated runs give identical results.
    """

    bbox: tuple[float, float, float, float] = DEFAULT_BBOX
    t0: str = "2024-07-27"
    t1: str = "2025-07-27"
    out_dir: Path = Path("outputs")
    cache_dir: Path = Path("data/cache")
    external_dir: Path = Path("data/external")
    concessions_path: Path | None = None
    resolution_m: float = 10.0
    max_cloud: float = DEFAULT_MAX_CLOUD
    search_window_days: int = SEARCH_WINDOW_DAYS
    min_area_ha: float = 0.5
    p_high: float = 0.7
    p_low: float = 0.3
    buffer_m: float = 500.0
    max_aoi_km2: float = 400.0
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError with a plain-English message if a setting is unusable."""
        w, s, e, n = self.bbox
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise ValueError(
                f"Bounding box {self.bbox} is not valid. Use west south east north "
                "in decimal degrees, with west < east and south < north."
            )
        if not (0 < self.p_low < self.p_high < 1):
            raise ValueError("Thresholds must satisfy 0 < p_low < p_high < 1.")
        if self.min_area_ha < 0:
            raise ValueError("min_area_ha cannot be negative.")
