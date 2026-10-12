"""Tests for scripts/fetch_datasets.py helpers.

Everything here uses tiny synthetic files written into a temp folder. No test
downloads a real dataset.
"""

import hashlib
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import fetch_datasets as fd  # noqa: E402


def test_human_size_units():
    assert fd.human_size(512) == "512.0 B"
    assert fd.human_size(2048) == "2.0 KB"
    assert fd.human_size(3 * 1024 * 1024) == "3.0 MB"


def test_sha256_known_answer(tmp_path):
    f = tmp_path / "abc.bin"
    f.write_bytes(b"abc")
    assert fd.sha256_of(f) == hashlib.sha256(b"abc").hexdigest()


def test_convert_mining_keeps_columns_and_reprojects(tmp_path):
    # Three squares in Web Mercator (EPSG:3857). The output must be EPSG:4326 and
    # keep only the three columns we use.
    src = tmp_path / "src.gpkg"
    gdf = gpd.GeoDataFrame(
        {"ISO3_CODE": ["PER", "GHA", "BRA"], "COUNTRY_NAME": ["Peru", "Ghana", "Brazil"],
         "AREA": [1.0, 2.0, 3.0], "EXTRA": ["x", "y", "z"]},
        geometry=[box(0, 0, 1000, 1000), box(2000, 0, 3000, 1000), box(4000, 0, 5000, 1000)],
        crs=3857,
    )
    gdf.to_file(src, driver="GPKG")
    dest = tmp_path / "out" / "mining.gpkg"
    count = fd.convert_mining(src, dest)
    assert count == 3
    out = gpd.read_file(dest)
    assert out.crs.to_epsg() == 4326
    assert set(out.columns) == {"ISO3_CODE", "COUNTRY_NAME", "AREA", "geometry"}
    assert list(out["ISO3_CODE"]) == ["PER", "GHA", "BRA"]
    # 1000 m east of the origin is about 0.009 degrees of longitude.
    assert 0.0 < out.geometry.iloc[0].bounds[2] < 0.02


def test_convert_mining_missing_column_is_clear_error(tmp_path):
    src = tmp_path / "bad.gpkg"
    gpd.GeoDataFrame({"NAME": ["x"]}, geometry=[box(0, 0, 1, 1)], crs=4326).to_file(src, driver="GPKG")
    with pytest.raises(fd.FetchError, match="expected columns"):
        fd.convert_mining(src, tmp_path / "out.gpkg")


def _make_openpit(folder: Path, pairs: int, unpaired_mask: bool = False) -> None:
    """Create empty placeholder files named like image/mask pairs."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(pairs):
        (folder / f"scene_{i}.tif").write_bytes(b"")
        (folder / f"scene_{i}_mask.tif").write_bytes(b"")
    if unpaired_mask:
        (folder / "orphan_mask.tif").write_bytes(b"")


def test_openpit_complete_folder_passes(tmp_path):
    folder = tmp_path / "Data"
    _make_openpit(folder, fd.OPENPIT_EXPECTED_PAIRS)
    problems, notes = fd.validate_openpit(folder)
    assert problems == []
    assert any("10 image files and 10 mask files" in n for n in notes)


def test_openpit_missing_pairs_is_reported(tmp_path):
    folder = tmp_path / "Data"
    _make_openpit(folder, 6)
    problems, _ = fd.validate_openpit(folder)
    assert any("Expected 10 mask files, found 6" in p for p in problems)


def test_openpit_orphan_mask_is_reported(tmp_path):
    folder = tmp_path / "Data"
    _make_openpit(folder, fd.OPENPIT_EXPECTED_PAIRS, unpaired_mask=True)
    problems, _ = fd.validate_openpit(folder)
    assert any("orphan_mask.tif" in p for p in problems)


def test_openpit_missing_folder_is_reported(tmp_path):
    problems, _ = fd.validate_openpit(tmp_path / "absent")
    assert problems and "Folder not found" in problems[0]


def _two_concessions(path: Path) -> Path:
    """One concession inside the test box and one far away."""
    gdf = gpd.GeoDataFrame(
        {"name": ["inside", "far"]},
        geometry=[box(-70.95, -12.95, -70.93, -12.93), box(-60.0, 5.0, -59.9, 5.1)],
        crs=4326,
    )
    gdf.to_file(path, driver="GeoJSON")
    return path


def test_concessions_clip_keeps_only_area(tmp_path):
    src = _two_concessions(tmp_path / "c.geojson")
    dest = tmp_path / "leases" / "concessions.geojson"
    count = fd.prepare_concessions(src, dest, aoi=(-71.0, -13.0, -70.8, -12.8))
    assert count == 1
    out = gpd.read_file(dest)
    assert list(out["name"]) == ["inside"]
    assert out.crs.to_epsg() == 4326


def test_concessions_without_aoi_keeps_everything(tmp_path):
    src = _two_concessions(tmp_path / "c.geojson")
    assert fd.prepare_concessions(src, tmp_path / "all.geojson", aoi=None) == 2


def test_concessions_bad_aoi_is_rejected(tmp_path):
    src = _two_concessions(tmp_path / "c.geojson")
    with pytest.raises(fd.FetchError, match="west < east"):
        fd.prepare_concessions(src, tmp_path / "x.geojson", aoi=(-70.0, -13.0, -71.0, -12.8))


def test_concessions_aoi_with_no_overlap_is_reported(tmp_path):
    src = _two_concessions(tmp_path / "c.geojson")
    with pytest.raises(fd.FetchError, match="No concession polygons"):
        fd.prepare_concessions(src, tmp_path / "x.geojson", aoi=(10.0, 10.0, 11.0, 11.0))


def test_concessions_unreadable_file_is_clear_error(tmp_path):
    bad = tmp_path / "not_geo.geojson"
    bad.write_text("this is not geospatial data", encoding="utf-8")
    with pytest.raises(fd.FetchError, match="Could not read"):
        fd.prepare_concessions(bad, tmp_path / "x.geojson", aoi=None)


def test_download_failure_gives_manual_steps(tmp_path):
    hint = "Manual steps: put the file in place by hand."
    with pytest.raises(fd.FetchError) as err:
        # Port 9 on localhost is almost always closed, so the connection fails at once.
        fd.download("http://127.0.0.1:9/file.gpkg", tmp_path / "file.gpkg", hint)
    assert hint in str(err.value)
    assert "resumes it" in str(err.value)
    assert not (tmp_path / "file.gpkg").exists()  # nothing half-written under the final name


def test_cli_parser_accepts_each_subcommand():
    parser = fd.build_parser()
    assert parser.parse_args(["mining"]).command == "mining"
    assert parser.parse_args(["openpit"]).command == "openpit"
    args = parser.parse_args(["concessions", "x.geojson", "--aoi", "-71", "-13", "-70.8", "-12.8"])
    assert args.aoi == [-71.0, -13.0, -70.8, -12.8]
    assert parser.parse_args(["all"]).command == "all"
