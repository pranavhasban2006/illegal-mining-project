"""Download and prepare the datasets the detector uses.

Run from the project folder:

    python scripts/fetch_datasets.py mining
    python scripts/fetch_datasets.py openpit
    python scripts/fetch_datasets.py concessions path_or_url [--aoi W S E N]
    python scripts/fetch_datasets.py all

Subcommands
-----------
mining        Global Mining Footprint polygons (Maus et al. 2022, PANGAEA doi 10.1594/PANGAEA.942325).
              Downloads the GeoPackage once, then writes a copy with the columns we use.
openpit       Open-Pit Mining Areas Dataset. It cannot be downloaded here, so this command tells
              you where to put it, then checks the files that are there.
concessions   A concessions file (GeoJSON, shapefile, zipped shapefile or GeoPackage) given as a
              local path or a URL. Optionally clipped to an area of interest.
all           Runs whatever can run and prints a checklist of what is present and what is missing.

Downloads resume if interrupted, and skip files that already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = REPO_DIR / "data" / "external"
MINING_DIR = EXTERNAL_DIR / "global_mining"
MINING_TARGET = MINING_DIR / "mining_polygons.gpkg"
MINING_URL = "https://download.pangaea.de/dataset/942325/files/global_mining_polygons_v2.gpkg"
MINING_LANDING = "https://doi.pangaea.de/10.1594/PANGAEA.942325"
MINING_COLUMNS = ["ISO3_CODE", "COUNTRY_NAME", "AREA"]  # plus geometry

OPENPIT_DIR = EXTERNAL_DIR / "open_pit_mining_dataset" / "Data"
OPENPIT_EXPECTED_PAIRS = 10
OPENPIT_SOURCE_NOTE = (
    "The Open-Pit Mining Areas Dataset is NOT included in this project and has no license "
    "file here. Cite the original authors when you use it. Look up the citation at the "
    "dataset's own page before publishing results."
)

CONCESSIONS_TARGET = REPO_DIR / "data" / "leases" / "concessions.geojson"
CONCESSIONS_SOURCE_DIR = REPO_DIR / "data" / "external" / "concessions_source"


class FetchError(RuntimeError):
    """A download or conversion failed. The message says what to do by hand."""


# ---------------------------------------------------------------- small helpers --

def human_size(num_bytes: float) -> str:
    """Return a size like '24.7 MB' for printing."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks so big files fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_bar(done: int, total: int | None, label: str) -> None:
    """Redraw a one-line text progress bar in place."""
    if total:
        frac = min(done / total, 1.0)
        filled = int(30 * frac)
        bar = "#" * filled + "-" * (30 - filled)
        text = f"\r{label} [{bar}] {frac * 100:5.1f}% {human_size(done)}/{human_size(total)}"
    else:
        text = f"\r{label} {human_size(done)}"
    sys.stdout.write(text)
    sys.stdout.flush()


def download(url: str, dest: Path, manual_hint: str) -> Path:
    """Download url to dest, resuming a partial file if one exists.

    Writes to dest.part first, then renames it, so an interrupted run never leaves a
    half-written file under the final name. Raises FetchError with manual steps on failure.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    start = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={start}-"} if start else {}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
            if resp.status_code == 416:  # the partial file is already complete
                part.replace(dest)
                return dest
            if resp.status_code not in (200, 206):
                raise FetchError(f"Server answered HTTP {resp.status_code} for {url}.\n{manual_hint}")
            if resp.status_code == 200:
                start = 0  # server ignored Range, so start again from zero
            total = resp.headers.get("Content-Length")
            total = int(total) + start if total else None
            mode = "ab" if start and resp.status_code == 206 else "wb"
            done = start
            last_draw = 0.0
            with part.open(mode) as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
                        done += len(chunk)
                        # Redraw at most about four times a second, so logs stay readable.
                        if time.time() - last_draw > 0.25:
                            print_bar(done, total, f"Downloading {dest.name}")
                            last_draw = time.time()
            print_bar(done, total, f"Downloading {dest.name}")
        print()
    except requests.RequestException as err:
        raise FetchError(
            f"Download failed: {err}\nThe partial file is kept, so running the command again "
            f"resumes it.\n{manual_hint}"
        ) from err
    part.replace(dest)
    return dest


# ------------------------------------------------------------------ mining ------

def convert_mining(src: Path, dest: Path) -> int:
    """Copy the mining GeoPackage to dest with only the columns we use, in EPSG:4326.

    Returns the number of polygons written. The source file has a single layer; this
    function reads that layer and keeps the geometry plus MINING_COLUMNS.
    """
    import geopandas as gpd

    gdf = gpd.read_file(src)
    missing = [c for c in MINING_COLUMNS if c not in gdf.columns]
    if missing:
        raise FetchError(
            f"The mining file does not have the expected columns: {missing}. "
            "It may be a different version. Check the dataset page at " + MINING_LANDING
        )
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)[MINING_COLUMNS + ["geometry"]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest, driver="GPKG", layer="mining_polygons")
    return len(gdf)


def cmd_mining(_args) -> int:
    """Download and convert the Global Mining Footprint polygons."""
    if MINING_TARGET.exists():
        print(f"Already present, skipping: {MINING_TARGET}")
        return 0
    manual = (
        "Manual steps:\n"
        f"  1. Open {MINING_LANDING} in a browser and download global_mining_polygons_v2.gpkg.\n"
        f"  2. Save it as {MINING_DIR / 'global_mining_polygons_v2.gpkg'}\n"
        "  3. Run this command again."
    )
    raw = MINING_DIR / "global_mining_polygons_v2.gpkg"
    try:
        if not raw.exists():
            download(MINING_URL, raw, manual)
        else:
            print(f"Raw file already downloaded: {raw}")
        print(f"SHA-256 of the raw file: {sha256_of(raw)}")
        print("(The dataset page does not publish a checksum, so record this value yourself.)")
        count = convert_mining(raw, MINING_TARGET)
    except FetchError as err:
        print(f"\nERROR: {err}")
        return 1
    print(f"Wrote {count} mining polygons to {MINING_TARGET}")
    print("License: CC BY-SA 4.0. Cite Maus et al. (2022), Global Mining Footprint, PANGAEA.")
    return 0


# ----------------------------------------------------------------- open-pit -----

def validate_openpit(folder: Path) -> tuple[list[str], list[str]]:
    """Check the Open-Pit folder. Returns (problems, notes).

    Expects OPENPIT_EXPECTED_PAIRS image/mask pairs. A file counts as a mask when its
    name contains 'mask'. Every mask must have an image with the same name apart from
    that word. The exact file naming of the source dataset was not checked here, so any
    mismatch is reported instead of silently accepted.
    """
    problems: list[str] = []
    notes: list[str] = []
    if not folder.is_dir():
        problems.append(f"Folder not found: {folder}")
        return problems, notes
    tifs = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff"))
    masks = [p for p in tifs if "mask" in p.stem.lower()]
    images = [p for p in tifs if p not in masks]
    notes.append(f"Found {len(images)} image files and {len(masks)} mask files.")

    def key(p: Path) -> str:
        return p.stem.lower().replace("mask", "").strip("_- ")

    image_keys = {key(p) for p in images}
    unpaired = [m.name for m in masks if key(m) not in image_keys]
    if unpaired:
        problems.append(f"Masks with no matching image: {unpaired}")
    if len(masks) != OPENPIT_EXPECTED_PAIRS:
        problems.append(f"Expected {OPENPIT_EXPECTED_PAIRS} mask files, found {len(masks)}.")
    return problems, notes


def cmd_openpit(_args) -> int:
    """Explain where the Open-Pit files go, then validate them if they are there."""
    print(OPENPIT_SOURCE_NOTE)
    print(f"\nPut the dataset's Data folder here:\n  {OPENPIT_DIR}")
    problems, notes = validate_openpit(OPENPIT_DIR)
    for n in notes:
        print(n)
    if problems:
        print("NOT READY:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("Open-Pit dataset looks complete.")
    return 0


# -------------------------------------------------------------- concessions -----

def prepare_concessions(source: Path, dest: Path, aoi: tuple[float, float, float, float] | None) -> int:
    """Read a concessions file, optionally clip it to aoi (W, S, E, N), and write GeoJSON.

    Returns the number of concession polygons written. The output is always EPSG:4326.
    """
    import geopandas as gpd
    from shapely.geometry import box

    try:
        gdf = gpd.read_file(source)
    except Exception as err:  # the driver error text is the most useful part
        raise FetchError(
            f"Could not read {source}: {err}\n"
            "Accepted formats: GeoJSON, shapefile (.shp, or a .zip of one), GeoPackage."
        ) from err
    if gdf.empty:
        raise FetchError(f"{source} has no features.")
    if gdf.crs is None:
        print("Warning: the file has no coordinate system. Assuming EPSG:4326 (lon/lat).")
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    if aoi is not None:
        w, s, e, n = aoi
        if not (w < e and s < n):
            raise FetchError("--aoi must be W S E N, with west < east and south < north.")
        gdf = gdf.clip(box(w, s, e, n))
    if gdf.empty:
        raise FetchError("No concession polygons fall inside the area you gave.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(dest, driver="GeoJSON")
    return len(gdf)


def cmd_concessions(args) -> int:
    """Fetch (if a URL) and convert a concessions file."""
    source_arg = args.source
    try:
        if source_arg.lower().startswith(("http://", "https://")):
            name = source_arg.rstrip("/").split("/")[-1].split("?")[0] or "concessions_download"
            source = download(source_arg, CONCESSIONS_SOURCE_DIR / name,
                              f"Download the file in a browser from {source_arg} and pass its local path.")
        else:
            source = Path(source_arg)
            if not source.is_file():
                raise FetchError(f"No file at {source}. Check the path, or give a URL.")
        aoi = tuple(args.aoi) if args.aoi else None
        count = prepare_concessions(source, CONCESSIONS_TARGET, aoi)
    except FetchError as err:
        print(f"ERROR: {err}")
        return 1
    print(f"Wrote {count} concession polygons to {CONCESSIONS_TARGET}")
    return 0


# ----------------------------------------------------------------- all ----------

def checklist() -> list[tuple[str, bool, str]]:
    """Return (name, present, detail) for every dataset the app can use."""
    rows = []
    if MINING_TARGET.exists():
        rows.append(("Global mining footprint", True, human_size(MINING_TARGET.stat().st_size)))
    else:
        rows.append(("Global mining footprint", False, "run: python scripts/fetch_datasets.py mining"))
    problems, _ = validate_openpit(OPENPIT_DIR)
    rows.append(("Open-Pit mining dataset", not problems,
                 "ready" if not problems else "see: python scripts/fetch_datasets.py openpit"))
    if CONCESSIONS_TARGET.exists():
        rows.append(("Concessions", True, human_size(CONCESSIONS_TARGET.stat().st_size)))
    else:
        rows.append(("Concessions", False,
                     "run: python scripts/fetch_datasets.py concessions <file or URL>"))
    return rows


def cmd_all(_args) -> int:
    """Run the mining download, check open-pit, then print the checklist."""
    cmd_mining(_args)
    print()
    cmd_openpit(_args)
    print("\nChecklist:")
    for name, present, detail in checklist():
        mark = "[x]" if present else "[ ]"
        print(f"  {mark} {name:<28} {detail}")
    return 0


# ----------------------------------------------------------------- main ---------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare the detector's datasets.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mining", help="download the Global Mining Footprint polygons").set_defaults(func=cmd_mining)
    sub.add_parser("openpit", help="check the Open-Pit dataset folder").set_defaults(func=cmd_openpit)
    p = sub.add_parser("concessions", help="convert a concessions file to data/leases")
    p.add_argument("source", help="local path or URL of a GeoJSON, shapefile, zip or GeoPackage")
    p.add_argument("--aoi", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="clip to this box (lon/lat)")
    p.set_defaults(func=cmd_concessions)
    sub.add_parser("all", help="run what is possible and print a checklist").set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.time()
    code = args.func(args)
    print(f"(finished in {time.time() - start:.1f} s)")
    return code


if __name__ == "__main__":
    sys.exit(main())
