"""O1: Download Sentinel-2 L2A band assets from CDSE for a search result item.

Primary path: direct S3 GetObject against CDSE's eodata bucket using S3
access-key credentials (generate at
https://eodata-s3keysmanager.dataspace.copernicus.eu). This only pulls the
~6 band files we need (tens of MB) instead of the full SAFE product zip
(800MB-1GB), which is far more reliable on CDSE's OData `$value` endpoint
that tends to truncate large streamed downloads.
"""

from __future__ import annotations

import os
import re
import time
import zipfile

import boto3
import requests

from .auth import CDSESession

S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
S3_BUCKET = "eodata"

# Bands used throughout this project (see indices.py), mapped to their
# native Sentinel-2 L2A resolution.
REQUIRED_BANDS = {
    "B02": "10m", "B03": "10m", "B04": "10m", "B08": "10m",
    "B11": "20m", "B12": "20m",
}


def _s3_client():
    access_key = os.environ.get("CDSE_S3_ACCESS_KEY")
    secret_key = os.environ.get("CDSE_S3_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError(
            "CDSE_S3_ACCESS_KEY / CDSE_S3_SECRET_KEY not found. Generate a key "
            "pair at https://eodata-s3keysmanager.dataspace.copernicus.eu"
        )
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="default",
    )


def download_bands(item, session: CDSESession | None, out_dir: str) -> dict:
    """Downloads required bands for a STAC item directly via S3. Returns {band: path}."""
    os.makedirs(out_dir, exist_ok=True)
    client = _s3_client()
    paths = {}

    for band, res in REQUIRED_BANDS.items():
        asset_key = f"{band}_{res}"
        asset = item.assets.get(asset_key)
        if asset is None:
            raise KeyError(f"Asset {asset_key} not found in item {item.id}")

        s3_key = asset.href.removeprefix(f"s3://{S3_BUCKET}/")
        out_path = os.path.join(out_dir, f"{band}_{res}.jp2")

        if not os.path.exists(out_path):
            client.download_file(S3_BUCKET, s3_key, out_path)

        paths[band] = out_path

    return paths


# --- Fallback: full-product OData download (kept for environments without
# S3 keys). Prone to truncating on large files; prefer download_bands above.

def download_product_zip(
    item, session: CDSESession, out_dir: str, max_retries: int = 5
) -> str:
    """Downloads the full SAFE .zip for a STAC item via OData. Returns zip path."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{item.id}.zip")
    tmp_path = out_path + ".part"

    product_asset = item.assets.get("Product")
    if product_asset is None:
        raise KeyError(f"No 'Product' asset found on item {item.id}")

    if os.path.exists(out_path):
        return out_path

    for attempt in range(max_retries):
        downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = session.auth_header()
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"

        try:
            resp = requests.get(
                product_asset.href, headers=headers, stream=True, timeout=600
            )
            if resp.status_code not in (200, 206):
                resp.raise_for_status()

            mode = "ab" if downloaded and resp.status_code == 206 else "wb"
            if mode == "wb":
                downloaded = 0
            with open(tmp_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

            os.rename(tmp_path, out_path)
            return out_path

        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as exc:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"Download interrupted ({exc.__class__.__name__}), "
                  f"resuming from {downloaded} bytes in {wait}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed to download {item.id} after {max_retries} attempts")


def extract_bands_from_zip(zip_path: str, out_dir: str) -> dict:
    """Extracts required band JP2s from a SAFE zip. Returns {band: path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for band, res in REQUIRED_BANDS.items():
            pattern = re.compile(rf"IMG_DATA/R{res}/.*_{band}_{res}\.jp2$")
            matches = [n for n in names if pattern.search(n)]
            if not matches:
                raise FileNotFoundError(
                    f"Band {band} ({res}) not found inside {zip_path}"
                )
            member = matches[0]
            out_path = os.path.join(out_dir, f"{band}_{res}.jp2")
            if not os.path.exists(out_path):
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
            paths[band] = out_path

    return paths
