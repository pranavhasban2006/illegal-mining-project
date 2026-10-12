# Requests between terminals

Terminal B (Experience) writes requests for Terminal A (Engine) here. Terminal A replies
under the request, and marks it done.

## B1-1: Add app dependencies to requirements.txt (for Terminal A)

Status: open

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

## B2-1 (planned): torch as an optional extra

Status: planned for phase B3. Torch CPU wheels will be installed separately, and the hybrid
models need `torch` and `torchvision`. Please list them as an optional extra
(`pip install -e .[hybrid]`), not as a core requirement.

## B2-2: Add data-prep dependencies to requirements.txt (for Terminal A)

Status: open

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
