# Interface contract (AUTHORITATIVE)

Owner: Terminal A (Engine). Changes must be written here and announced in docs/REQUESTS.md
before either terminal relies on them.

## 1. Python API

```python
mining_detector.pipeline.run_demo(out_dir="outputs") -> RunResult
mining_detector.pipeline.run_live(aoi_bbox, t0_date, t1_date, out_dir="outputs",
        concessions_path=None, progress=None, max_cloud=20.0,
        max_aoi_km2=400.0, hard_cases_path=None) -> RunResult
```

`progress` is an optional callable `progress(fraction: float, message: str)`, with
fraction between 0 and 1 and a plain-English message.

`RunResult` (dataclass): `alerts_path: Path`, `report_path: Path`, `summary: dict`,
`metrics: dict`.

## 2. Files written to out_dir

- **alerts.geojson**: FeatureCollection, CRS EPSG:4326. Polygon features with properties
  `{id: int, legal_status: "legal"|"disputed"|"illegal", confidence: float 0-1,
  area_ha: float, detected_on: "YYYY-MM-DD"}`.
- **report.json**: `{aoi_bbox: [w,s,e,n], t0, t1, n_alerts, total_new_disturbance_ha,
  by_legal_status: {legal, disputed, illegal (ha)}, model: {best, candidates:
  [{name, f1_train, f1_test, overfit_flag}], feature_importance: {name: float}},
  data_sources: [str], caveats: [str]}`. Optional key `validation` (added in A4).
- **change_map.png**, **probability_t1.png**: quicklook images for the UI.

## 3. CLI

```
python -m mining_detector demo
python -m mining_detector live --bbox W S E N --t0 YYYY-MM-DD --t1 YYYY-MM-DD [--concessions PATH]
python -m mining_detector live --example
python -m mining_detector fetch [--bbox W S E N] [--date YYYY-MM-DD]
```

## 4. Dataset locations (gitignored)

- `data/cache/`: Sentinel-2 band cache, keyed by item id and bbox.
- `data/external/`: downloaded public datasets.
- `data/leases/`: concession GeoJSON.
- `data/external/global_mining/mining_polygons.gpkg`: Global Mining Footprint polygons
  (Maus et al. 2022), used as positive labels. Terminal B's `scripts/fetch_datasets.py mining`
  writes this file.

## 5. Labels and licensing rules

- Labels never come from the classifier's own features. Positives come from the mining
  footprint file or from concessions; negatives are dense vegetation (NDVI > 0.6) more
  than 300 m from any positive, plus permanent water.
- `legal`: inside an active concession. `disputed`: touches an inactive or expired
  concession, or lies within `buffer_m` (default 500 m) of any concession. `illegal`: otherwise.
- New disturbance = P(mining, t1) high AND P(mining, t0) low, then morphological cleanup
  and a minimum area of 0.5 ha.
- The default date pair compares the same calendar date one year apart, to avoid a
  seasonal phenology confound.

## 5b. Additions in phases A2 to A4

- `report.json` also contains `validation` when check points are given:
  `{n_points, n_scored, n_outside_grid, n_unsure_dropped, confusion: {tp,fp,tn,fn},
  precision, recall, f1, caveat}`.
- `model.candidates[i]` also has `precision_test` and `recall_test`.
  `model.label_counts`, `model.split`, `model.training_rows` and `model.comparison` are added.
- Extra CLI flags for `live`: `--validation PATH`, `--review-chips N`, `--hard-cases PATH`,
  `--eval-hard-cases` (with `--hard-cases`, runs the pipeline twice and writes
  `hard_case_eval/hard_case_holdout.json`), `--max-aoi-km2`, `--max-cloud`, `--example`.
- `--t0` is optional for `live`: it defaults to the same calendar date one year before `--t1`.
- `outputs/chips/` holds review chip PNGs and `review_template.csv`
  (chip_id, category, file, lon, lat, label, notes). Label values are 0, 1 or `unsure`.
- Check-point files (`--validation`) are CSV with lon, lat, label, or GeoJSON with a
  `label` property. Rows marked `unsure` are ignored.
- Hard-case files (`--hard-cases`) use the same format. Each point overrides the labels in a
  neighbourhood (0 px by default).

## 5c. Default example area

`DEFAULT_BBOX = (-72.10, -14.45, -71.95, -14.30)`, a gold-mining cluster near Cusco.
The earlier La Pampa box has no usable Global Mining Footprint polygons, so it cannot
supply training labels. Mining polygons over 100 km2 are dropped as data errors.

## 6. Quality rules

- Every engine module has pytest tests on synthetic data with a known answer.
- Network tests carry `@pytest.mark.network` and are skipped by default.
- Demo output is labelled as synthetic in both the caveats and the metrics.
