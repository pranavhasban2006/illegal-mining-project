# Illegal Mining Detector — Two-Terminal Build Plan

Build the project from scratch, fast, and easy to use. Two Claude Code sessions run side by side:

| Terminal | Role | Owns (only edits these) |
|---|---|---|
| **A — Engine** | data → features → detection → change → legal status → alerts | `src/mining_detector/*.py` (except `hybrid_models/`), `tests/engine/` |
| **B — Experience** | friendly app, dataset downloaders, deep models, docs, one-click scripts | `app/`, `scripts/`, `src/mining_detector/hybrid_models/`, `tests/experience/`, `README.md`, `docs/` |

**Rule: never edit the other terminal's files.** That is what makes side-by-side safe.

## Setup (do once, 2 minutes)

```powershell
mkdir $HOME\illegal-mining-detector; cd $HOME\illegal-mining-detector
git init
py -3.11 -m venv .venv
```
Open two terminals in that folder and start `claude` in each. Paste the **Shared Contract** below as the first message in BOTH, then run phases in order. Each phase ends with "reply DONE + how to run it" — when both terminals have finished the same-numbered phase, move on (sync points marked ⏸).

## What we changed vs. your original plan (to save time)

- **No CDSE credentials needed.** Sentinel-2 L2A comes from free, anonymous STAC catalogs (Element84 Earth Search on AWS: `https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`; fallback Microsoft Planetary Computer). Reads are windowed COGs, so only the bands/AOI needed are fetched. CDSE stays as an optional backend.
- **Public datasets replace hand labeling:**
  - *Global Mining Footprint* (Maus et al. 2022, PANGAEA, ~74k mining polygons worldwide) → positive labels, any AOI. Verify the download link on PANGAEA before use.
  - *ESA WorldCover / high-NDVI forest* → negative labels.
  - *RAISG concessions* (Peru/Amazon) → optional legal/illegal layer; app also accepts any user-uploaded GeoJSON.
  - *Open-Pit Mining Areas Dataset* (the 10 image/mask TIFF pairs you already have) → deep-model segmentation. No license file, so cite its source.
  - *Copernicus DEM GLO-30* (public AWS) → optional terrain; your own testing showed it adds ~nothing, so it's off by default.
- Skipped on purpose (documented as future work): GAN/CLAHE track, QC-SMOTE, EuroMineNet/OSCD downloads, multi-labeler tool.

---

## SHARED CONTRACT (paste first in BOTH terminals)

```
We are building "Illegal Mining Detector": a Sentinel-2 pipeline that finds new surface-mining
disturbance between two dates, classifies each site as legal / disputed / illegal against
concession polygons, and exports geo-referenced alerts. Priorities: (1) works end-to-end within
hours, (2) extremely easy for a non-expert to run, (3) honest metrics (no self-referential labels).
Windows 11, PowerShell, Python 3.11, venv at .venv. Package: src/mining_detector (use PYTHONPATH=src
or `pip install -e .`).

TWO CLAUDE SESSIONS WORK IN PARALLEL ON THIS REPO. You are told below which one you are.
Only edit the files your terminal owns. Never run `git add -A`; only `git add` your own paths.

OWNERSHIP
- Terminal A (Engine): src/mining_detector/*.py except hybrid_models/, tests/engine/, pyproject.toml,
  requirements.txt (A is the only one who edits these two; B asks A via a note in docs/REQUESTS.md).
- Terminal B (Experience): app/, scripts/, src/mining_detector/hybrid_models/, tests/experience/,
  README.md, docs/ (except docs/CONTRACT.md which A creates).

INTERFACE CONTRACT (do not change without writing it to docs/CONTRACT.md and docs/REQUESTS.md)
1. mining_detector.pipeline.run_demo(out_dir="outputs") -> RunResult
   mining_detector.pipeline.run_live(aoi_bbox, t0_date, t1_date, out_dir="outputs",
        concessions_path=None, progress=None) -> RunResult
   `progress` is an optional callable progress(fraction: float, message: str).
   RunResult is a dataclass: alerts_path, report_path, summary (dict), metrics (dict).
2. Files written to out_dir:
   - alerts.geojson : FeatureCollection, CRS EPSG:4326, Polygon features with properties
       {id:int, legal_status: "legal"|"disputed"|"illegal", confidence: float 0-1, area_ha: float,
        detected_on: "YYYY-MM-DD"}
   - report.json : {aoi_bbox:[w,s,e,n], t0, t1, n_alerts, total_new_disturbance_ha,
        by_legal_status:{legal:ha, disputed:ha, illegal:ha},
        model:{best:str, candidates:[{name,f1_train,f1_test,overfit_flag}], feature_importance:{name:float}},
        data_sources:[str], caveats:[str]}
   - change_map.png and probability_t1.png (quicklook images for the UI)
3. CLI: `python -m mining_detector demo`  and
        `python -m mining_detector live --bbox W S E N --t0 YYYY-MM-DD --t1 YYYY-MM-DD [--concessions path]`
4. Dataset locations (gitignored): data/cache/ (STAC band cache), data/external/ (downloaded datasets),
   data/leases/ (concession GeoJSON).
5. Until Terminal A's demo works, Terminal B develops against fixtures that B generates itself
   in tests/experience/fixtures/ (same schema as above).

QUALITY BAR: every module has at least one pytest test using synthetic data with a known answer.
Plain-English docstrings. No placeholder/mock results presented as real. If something can't be run
here (no network, no GPU), say so and do the part you can verify.
Reply "ACK" and wait for the phase prompt.
```

---

# TERMINAL A — ENGINE

### A1 — Scaffold + synthetic demo end-to-end (target: ~30–45 min)

```
You are Terminal A (Engine). Phase A1.
1. Create the project scaffold: pyproject.toml (installable, name mining-detector, src layout),
   requirements.txt (numpy, scipy, scikit-learn, imbalanced-learn, rasterio, shapely, geopandas, pyproj,
   pystac-client, stackstac OR rioxarray, matplotlib, pillow, pytest, tqdm, python-dotenv),
   .gitignore (.venv, data/cache, data/external, outputs, __pycache__), src/mining_detector/__init__.py,
   __main__.py (CLI per contract, argparse, friendly --help).
2. Implement, each with tests in tests/engine/:
   - config.py: dataclass Config (aoi bbox, dates, bands, thresholds, paths); sensible Peru/La Pampa default
     bbox as an example, but everything overridable.
   - indices.py: NDVI, NDMI, BSI, NDWI, plus compute_feature_stack(bands)->(H,W,F) with named features.
   - change_detection.py: bi-temporal change: per-scene normalization of features, delta features,
     new-disturbance mask, morphological cleanup, min-area filter.
   - licensing.py: classify each disturbance polygon vs concession polygons: inside active concession
     = "legal", overlapping an inactive/expired concession or within a buffer = "disputed", else "illegal".
   - alerts.py: raster mask -> polygons (rasterio.features.shapes) -> GeoJSON per contract, area_ha in UTM.
   - synthetic.py: generates a synthetic 2-date scene (one existing licensed mine, two new unlicensed
     sites, vegetation + river + noise) with a matching concessions GeoDataFrame.
   - pipeline.py: run_demo(out_dir) wiring everything; writes alerts.geojson, report.json,
     change_map.png, probability_t1.png exactly per contract. For the demo, use a simple spectral
     disturbance score (no trained model yet).
3. Write docs/CONTRACT.md (copy the contract, mark it authoritative).
4. Run: pytest tests/engine -q  and  `python -m mining_detector demo`. Fix until green. Show the
   alerts summary. `git add` only your paths and commit "A1: engine scaffold + demo".
Reply DONE + exact commands to run the demo.
```

### A2 — Real Sentinel-2 via STAC + public labels (target: ~45 min)

```
Terminal A, Phase A2: real data, no credentials.
1. data_sources.py: search_scenes(bbox, date, max_cloud=20) using pystac-client against
   https://earth-search.aws.element84.com/v1 (collection sentinel-2-l2a); auto-fallback to Microsoft
   Planetary Computer (planetary-computer package signing) if it fails. Pick the least-cloudy scene
   within +/- 15 days of the requested date.
2. preprocess.py: windowed read (no full-tile download) of B02,B03,B04,B08,B11,B12 and SCL, cropped to
   the AOI, resampled to 10 m, reflectance scaled, cloud/shadow/water-as-needed masked via SCL, cached
   under data/cache/ keyed by item id + bbox. Both dates must be co-registered on the same grid.
   Raise clear, human-readable errors (e.g. "no scene under 20% cloud within 15 days of 2026-07-27;
   try --max-cloud 40").
3. labels.py: build training labels from PUBLIC data, never from the classifier's own features:
   - positives: Global Mining Footprint polygons (Maus et al. 2022) clipped to the AOI. Read from
     data/external/global_mining/ if present (Terminal B ships a downloader; define the expected filename
     in docs/CONTRACT.md: data/external/global_mining/mining_polygons.gpkg). If absent, fall back to
     concession polygons from concessions_path if given, else raise a clear message telling the user
     to run `python scripts/fetch_datasets.py mining`.
   - negatives: pixels >300 m from any positive polygon with NDVI > 0.6 (dense vegetation) plus permanent
     water; record counts of each.
   - an optional extra-labels hook for hand-labeled points (GeoJSON with label 0/1) that overrides.
4. Tests with synthetic rasters/polygons for labels.py and the masking logic (no network in tests;
   mark network tests with @pytest.mark.network and skip by default).
5. Add `python -m mining_detector fetch --bbox ... --date ...` to just download+cache a scene and print
   its size, cloud %, and a quicklook PNG path — a quick smoke test of connectivity.
Run the fetch command for the default AOI to prove it works over the network; report what happened
honestly (success/failure/what you saw). Commit your own paths. Reply DONE.
```

### A3 — Trained detector + model selection + live run (target: ~45 min)

```
Terminal A, Phase A3.
1. model_selection.py: per-pixel table from the feature stack + labels; stratified subsample to a
   capped size (e.g. 200k px) for speed; holdout split 80:20 (option for 10-fold CV); SMOTE on the
   TRAIN portion only; candidates: LogisticRegression L1, L2, RandomForest, SVM-RBF (on a smaller
   subsample), GradientBoosting with early stopping; metrics precision/recall/F1 on train and test;
   overfit flag when train-test F1 gap > 0.05; select by test F1; return feature importances (RF).
   Use spatial blocking for the split (split by AOI tiles, not random pixels) to avoid leakage —
   report both if cheap.
2. detection.py: MiningDetector.fit / predict_proba_map with per-scene StandardScaler re-fit
   (normalize_per_scene=True) so t0 and t1 are comparable.
3. Update change_detection.py + pipeline.run_live(): compare the SAME calendar date one year apart
   by default (document why: seasonal phenology confound), new disturbance = P(mining,t1) high AND
   P(mining,t0) low, min-area filter, then licensing + alerts per contract. Call progress(...) at each
   step with a plain-English message. Fill report.json fully incl. model candidates, feature
   importance, data_sources, and auto-generated `caveats` (e.g. "labels come from <dataset>",
   "near-perfect F1 on obvious cases does not imply accuracy on ambiguous land").
4. CLI `live` works per contract. Add `python -m mining_detector live --example` that runs the
   default AOI/dates.
5. Tests: model_selection on synthetic separable data (expect F1>0.9 and no leakage between train/test),
   SMOTE only on train, per-scene normalization test (shifted+scaled copy of a scene yields ~same
   probability map).
Run live on the default AOI and report actual numbers and runtime. If the network blocks you, say so,
and verify the rest with the synthetic path. Commit your own paths. Reply DONE.
```

### A4 — Honest validation + robustness (target: ~30 min)

```
Terminal A, Phase A4.
1. validation.py: given a GeoJSON/CSV of hand-labeled check points (lon, lat, label 0/1/unsure) compute
   confusion matrix, precision, recall, F1 for the pipeline's t1 mask; write into report.json under
   `validation`. Include a `make_review_chips(n=40)` helper that exports small stratified chip PNGs
   (near-boundary, uncertain-probability, bare-far-from-mine, mid-NDVI) + a CSV template for labeling
   — Terminal B's app will use these.
2. Add `--hard-cases path.csv` to fold human labels into training (9x9 px neighborhood override) and
   an evaluation mode that holds out half to measure the improvement, as in the original design.
3. Robustness: empty-alert case produces valid empty FeatureCollection + report; AOI too large
   (> configurable max km^2) gives a friendly error; deterministic seeds; runtime logging.
4. Update docs/CONTRACT.md with anything added. Run full pytest tests/engine; fix failures; commit.
5. Print a short summary of final capabilities and known limitations for the README (Terminal B writes
   the README; put your text in docs/ENGINE_NOTES.md).
Reply DONE.
```

---

# TERMINAL B — EXPERIENCE

### B1 — Friendly app against fixtures (target: ~40 min, starts immediately)

```
You are Terminal B (Experience). Phase B1. Do not wait for Terminal A.
1. Add to your own paths only. Create tests/experience/fixtures/ with a generator script that writes a
   realistic alerts.geojson + report.json + change_map.png + probability_t1.png following the contract
   exactly (a dozen small polygons near La Pampa, Peru: lon -70.9, lat -12.9; mixed legal/disputed/
   illegal; report.json fully populated).
2. Build app/ — a Streamlit app (app/Home.py + app/pages/) designed for a complete beginner:
   - Home: one-paragraph plain-English explanation, a big "Try the demo" button, then "Analyze my area".
   - Results page: interactive map (folium via streamlit-folium; OpenStreetMap + satellite basemap
     toggle), colored polygons (illegal=red, disputed=orange, legal=green), click for confidence/area,
     summary metric cards (number of alerts, hectares by status), sortable table, filter by status and
     confidence slider, download buttons (GeoJSON, CSV, a one-page HTML/PDF-ish report).
   - Every technical term has a "?" tooltip with a plain-English explanation.
   - A sidebar showing the data folder the app reads (default outputs/), with a selector for older runs.
3. The app loads whatever is in outputs/ (or the fixture folder if outputs/ is empty, with a visible
   banner "Showing SAMPLE data").
4. Write scripts/run_app.ps1 (activates venv, installs missing deps, launches streamlit) and a double-
   clickable scripts/run_app.bat. Add app deps to docs/REQUESTS.md for Terminal A to add to
   requirements.txt (do not edit requirements.txt).
5. Tests: tests/experience/test_loaders.py for the app's loader/summary functions (keep logic in
   app/lib/ so it's testable without Streamlit). Run them. Launch the app headlessly, confirm it starts
   (curl localhost:8501 health endpoint), then stop it.
Commit your own paths. Reply DONE + how to launch.
```

### B2 — Dataset downloaders + one-click setup (target: ~30 min)

```
Terminal B, Phase B2.
1. scripts/fetch_datasets.py with subcommands (argparse, progress bars, resumable, checksum/size print,
   skips if present, clear errors with manual-download instructions if a link fails):
   - `mining`: Global Mining Footprint polygons (Maus et al. 2022; PANGAEA doi 10.1594/PANGAEA.942325).
     Find the actual current download link, verify it responds before coding it, convert to
     data/external/global_mining/mining_polygons.gpkg (EPSG:4326, only needed columns). If direct download
     isn't possible, print exact manual steps and the expected target path.
   - `openpit`: tells the user where the Open-Pit Mining Areas Dataset must be placed
     (data/external/open_pit_mining_dataset/Data/), validates the 10 image/mask pairs if present.
     Note in output: no license file included, cite the original source.
   - `concessions`: accepts a path/URL to a concessions GeoJSON/shapefile (e.g. RAISG) and clips/converts
     to data/leases/concessions.geojson; `--aoi W S E N` to clip.
   - `all`: runs what's possible and prints a checklist of what is present/missing.
2. scripts/setup.ps1: creates .venv with Python 3.11 if missing, installs requirements, runs fetch for
   `mining`, runs `pytest -q`, prints "Setup complete — next: scripts\run_app.bat". Idempotent.
3. app/pages/Datasets.py: shows a checklist (present/missing, size) for each dataset with the exact
   command/button to fetch the missing ones (button calls the script via subprocess and streams logs).
4. Tests for path validation/conversion helpers using tiny synthetic files. Do NOT download huge files
   in tests. Run what you can; report honestly which downloads you verified over the network and which
   you could not.
Commit your own paths. Reply DONE.
```

### B3 — Hybrid deep models on the Open-Pit dataset (target: ~45 min)

```
Terminal B, Phase B3. Work only under src/mining_detector/hybrid_models/ plus scripts/ and app/.
1. Install CPU torch (pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu)
   — tell Terminal A via docs/REQUESTS.md to list torch as an optional extra (don't edit
   requirements.txt yourself).
2. Implement models.py with five real PyTorch architectures, small (<50k params) so they train on CPU in
   minutes: UNetTransformer (segmentation), ResNetAttentionUNet (segmentation, with real attention
   gates), YoloViTLSTM (patch classification), CNNBiLSTM (patch classification), SiameseCNNTransformer
   (bi-temporal change classification). Include a `lite=True` variant of each (half width, single
   attention/LSTM layer).
3. openpit_dataset.py: tile the 10 image/mask TIFF pairs into 64x64 patches, hold out 2 whole source
   tiles for test (no spatial leakage), normalize, light augmentation (flips/rot90) on train only.
4. train_openpit.py: trains the four non-Siamese models (class-weighted loss to avoid majority
   collapse; early stopping), logs IoU/Dice for segmentation and acc/F1 for classification, saves
   data/hybrid_models_results/openpit_results.json and model weights. Siamese: train on synthetic
   bi-temporal pairs from mining_detector.synthetic if available else document it as demo-only.
   Add benchmark_lite.py (params, CPU latency, metric for full vs lite).
5. app/pages/Models.py: table + bar charts of results, plus a "Try it" widget: upload a satellite image
   crop -> choose model -> see predicted mask overlay. Include a visible honesty panel: test patches come
   from only 2 source scenes, so this measures interpolation, not generalization to new places.
6. If the Open-Pit dataset isn't on disk, make everything degrade gracefully (clear message, tests still
   pass using a tiny synthetic stand-in). Tests: shape tests for every model forward pass, one-step
   training loss decreases, patch tiling correctness.
Run training for real if the dataset is present (cap epochs so it finishes in <15 min) and report real
numbers; otherwise say that you could not. Commit your own paths. Reply DONE.
```

### ⏸ SYNC POINT — wait until Terminal A reports A3 DONE, then:

### B4 — Wire live runs + docs + polish (target: ~45 min)

```
Terminal B, Phase B4. Terminal A's engine is now available — read docs/CONTRACT.md first.
1. app/pages/Analyze.py: step-by-step wizard for non-experts:
   Step 1 Pick area (a map with draw-rectangle via folium Draw, OR paste coordinates, OR choose preset
   "La Pampa, Peru" / "Example: Ghana galamsey" with their known bboxes; show area in km^2 and warn
   if over the engine's limit).
   Step 2 Pick dates (default: same calendar date one year apart, with a plain-English note explaining
   why).
   Step 3 Optional concessions upload (GeoJSON) — explain "without this, every alert is 'illegal'".
   Step 4 Run: calls mining_detector.pipeline.run_live(...) in a background thread with the progress
   callback driving st.progress + plain-English status lines; on success jump to Results; on error
   show the engine's friendly message plus a "what to try" suggestion.
2. Results page upgrades: show change_map.png and probability_t1.png side by side, a "Model & trust"
   expander (candidates, F1, overfit flags, feature importance, caveats from report.json), and a
   prominent caveats box. Never hide the caveats.
3. Add a "Review" page (optional but cheap): shows engine-exported review chips and lets the user label
   them mining / not mining / unsure, saving to CSV in the format the engine's validation.py expects
   (check docs/CONTRACT.md or docs/ENGINE_NOTES.md).
4. README.md: 60-second quick start (setup.ps1 → run_app.bat), screenshot placeholders, what each page
   does, how to use your own area, FAQ, honest limitations (adapt from docs/ENGINE_NOTES.md), data
   sources & licenses/citations table, project objectives O1–O5 mapped to modules. docs/USER_GUIDE.md
   for non-technical users (glossary: NDVI, NDMI, BSI, NDWI, SMOTE, F1, concession).
5. End-to-end check: run the app, run the demo through the UI path, and if the network works run the
   live example; confirm results render. Run full pytest (both test folders). Fix issues in your own
   files; log engine problems to docs/REQUESTS.md for Terminal A rather than editing engine files.
Commit your own paths. Reply DONE with a short list of anything still broken.
```

---

## Final integration (either terminal, 10 min)

```
Run: powershell scripts\setup.ps1; pytest -q; python -m mining_detector demo; scripts\run_app.bat.
Resolve anything left in docs/REQUESTS.md, make sure README quick start works from a fresh clone
(new venv), and tag git v0.1.
```

## Tips

- If one terminal finishes early, paste the next phase; only the ⏸ marker needs both.
- If the network is slow, A2/A3 and B2 are the only phases that need it; everything else works offline on synthetic data.
- Keep the honest-caveats language: your original README's biggest strength is that it reports negative and ceiling-effect results plainly. Both terminals are told to keep that.
