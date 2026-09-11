"""O1: Search CDSE's STAC catalog for Sentinel-2 L2A scenes over an AOI."""

from __future__ import annotations

from pystac_client import Client

STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
COLLECTION = "sentinel-2-l2a"


def search_scenes(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    max_cloud_cover: float = 20.0,
    limit: int = 20,
):
    """bbox in WGS84 (minx, miny, maxx, maxy). Dates as 'YYYY-MM-DD'.

    Returns a list of STAC items sorted by datetime, filtered to L2A
    products under max_cloud_cover.
    """
    client = Client.open(STAC_URL)
    search = client.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        max_items=limit,
        query=[f"eo:cloud_cover<{max_cloud_cover}"],
    )
    items = list(search.item_collection())
    items.sort(key=lambda i: i.datetime)
    return items
