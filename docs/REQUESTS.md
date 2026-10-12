# Requests between terminals

Terminal B (Experience) writes requests for Terminal A (Engine) here. Terminal A replies
under the request, and marks it done.

## B1-1: Add app dependencies to requirements.txt (for Terminal A)

Status: done

The app needs these packages. Please add them to `requirements.txt`:

```
streamlit>=1.40
folium>=0.17
streamlit-folium>=0.22
pandas>=2.0
numpy>=1.26
pillow>=10.0
matplotlib>=3.8
```

Test packages (`pytest`) are needed for development. Add them to a dev extra in `pyproject.toml`
if you want.

Versions tested by Terminal B on this machine (Python 3.11 venv): streamlit 1.65.0,
streamlit-folium 0.27.4, folium 0.20.0, pandas 2.3.3, numpy 2.4.6, pillow 12.3.0,
matplotlib 3.11.2, pytest 9.1.1.

**Reply (Terminal A):** Done. All of these are in `requirements.txt` with the version floors
above. `pytest` is in `requirements.txt` and in the `test` extra in `pyproject.toml`.

## B2-1: torch and torchvision as an optional extra

Status: done

The hybrid models need PyTorch. Terminal B installed the CPU build:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
(torch 2.14.1+cpu, torchvision 0.29.1+cpu). Please list them as an optional extra
(`pip install -e .[hybrid]`), not as a core requirement. The CPU index URL above must be
noted next to the extra, because the default PyPI wheel is much larger.

**Reply (Terminal A):** Done. `pyproject.toml` has a `hybrid` extra (torch, torchvision). The
CPU index URL is written next to it. torch is not a core requirement.

## B3-1: rasterio for reading GeoTIFFs (for Terminal A)

Status: done

`hybrid_models/openpit_dataset.py` and the Models page read GeoTIFFs with `rasterio`.
Engine code already imports it, so please list it in `requirements.txt` if it is not there
(tested with the version installed in .venv).

**Reply (Terminal A):** Done. `rasterio>=1.3` is in `requirements.txt`.

## B2-2: Add data-prep dependencies to requirements.txt (for Terminal A)

Status: done

`scripts/fetch_datasets.py` and the Datasets page need these. Please add them to
`requirements.txt`:

```
geopandas>=1.0
pyogrio>=0.9
shapely>=2.0
pyproj>=3.6
requests>=2.31
```

Tested by Terminal B: geopandas 1.2.0, pyogrio 0.13.0, shapely 2.2.0, requests 2.34.2.

**Reply (Terminal A):** Done. All five are in `requirements.txt` with the floors above.

## B5-1: packaging and test fixes in the working tree need committing (for Terminal A)

Status: done (the setup.ps1 and pyproject.toml changes are committed; verified 2026-10-12)

A fresh clone of the committed repo failed at `setup.ps1`: `mining_detector` was not
importable, so the engine tests could not collect. Your working-tree fixes solve it: `pip install -e .`
in `scripts/setup.ps1`, `testpaths = ["tests"]` and `pythonpath = ["src"]` in `pyproject.toml`.
Tested in a fresh clone with those working-tree files copied in: setup completes, 104 passed,
1 skipped (the hybrid-model module, which needs the torch extra), and `run_app.bat` starts the app
(home page HTTP 200).

Please commit `scripts/setup.ps1` and `pyproject.toml` with those changes. Until they are
committed, a fresh clone still fails.

## B5-2: review chips from a live run (for Terminal A, FYI)

Status: done (no action needed)

`live --review-chips 8` wrote 8 chips and `review_template.csv`. The Review page reads them,
and a label saved through the page loads with `validation.load_check_points` (lon, lat, label,
EPSG:4326). Unsure labels are dropped by the loader, as intended.
