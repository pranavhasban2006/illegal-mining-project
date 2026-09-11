import geopandas as gpd
from shapely.geometry import box
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mining_detector import real_labels


def _concessions_gdf():
    return gpd.GeoDataFrame(
        {
            "situacion": ["Titulada", "Explotación", "Bloqueado", "Extinguido", "Trámite"],
            "geometry": [box(i, 0, i + 1, 1) for i in range(5)],
        },
        crs="EPSG:4326",
    )


def test_active_concessions_keeps_only_active_statuses():
    gdf = _concessions_gdf()
    active = real_labels.active_concessions(gdf)
    assert set(active["situacion"]) == {"Titulada", "Explotación"}
    assert len(active) == 2


def test_active_concessions_excludes_inactive_statuses():
    gdf = _concessions_gdf()
    active = real_labels.active_concessions(gdf)
    assert "Bloqueado" not in set(active["situacion"])
    assert "Extinguido" not in set(active["situacion"])
    assert "Trámite" not in set(active["situacion"])


def test_active_concessions_passthrough_without_situacion_column():
    gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    result = real_labels.active_concessions(gdf)
    assert len(result) == 1
