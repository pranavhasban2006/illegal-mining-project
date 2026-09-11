# Illegal Mining Detector

Sentinel-2-driven pipeline for detecting illegal/unregulated surface mining.
Maps to project objectives O1-O5:

| Stage | Objective | Module |
|---|---|---|
| O1 | Data acquisition (CDSE Sentinel-2 L2A) | `auth.py`, `search.py`, `download.py`, `preprocess.py` |
| O2 | Mining-area detection (spectral features + RF classifier) | `indices.py`, `detection.py` |
| O3 | Bi-temporal change detection | `change_detection.py` |
| O4 | Licensed vs. unlicensed footprint discrimination | `licensing.py` |
| O5 | Geo-referenced alert export (GeoJSON) | `alerts.py` |

`pipeline.py` orchestrates all five stages.

## Known limitations (real-data run, Madre de Dios pilot)

Two methodological issues surfaced running O1-O5 against real Sentinel-2
data over La Pampa, Peru, both fixed but worth documenting for the review:

1. **Cross-date scaling.** Two Sentinel-2 scenes from different dates differ
   in atmosphere/sun-angle beyond any real land-cover change. Applying one
   date's fitted feature scaler to another date's raw reflectance
   miscalibrates the classifier. Fix: `detection.py`'s `predict_proba_map`
   re-fits a fresh `StandardScaler` on each scene's own statistics
   (`normalize_per_scene=True`), so both dates are compared in relative
   terms rather than one date's absolute scale.

2. **Seasonal phenology confound.** Initially compared June 2026 vs. August
   2026 (two dates within the same Amazon dry-season progression). Verified
   directly: scene-wide NDVI dropped 0.48->0.32 and NDMI 0.19->0.12 across
   the *entire* AOI between those dates, with no plausible mining-driven
   cause -- vegetation naturally dries out through the dry season,
   mimicking a bare-soil signature at scale. This alone inflated "new
   disturbance" to an implausible ~18,000-28,000 ha in two months. Fix:
   compare the **same calendar date one year apart** instead
   (`config.py`: `2025-07-27` vs `2026-07-27`), which matches sun
   angle/season/phenology so a remaining reflectance shift is far more
   likely to reflect genuine change. This reduced the detected area to
   ~8,500 ha/year -- still plausible for this specific hotspot (MAAP
   documents historical losses of low-thousands of ha/year at La Pampa) but
   likely still somewhat overestimated.

3. **Weak-label ceiling -- fixed via real_labels.py.** `weak_labels.py`
   originally bootstrapped training labels from percentile thresholds on
   the *same* spectral features the classifier then trained on, so every
   algorithm scored a self-referential f1=1.000: the classifier was just
   re-deriving its own label-generation rule, not demonstrating real-world
   accuracy. Fixed by sourcing real ground truth externally instead:
   `real_labels.py` uses RAISG's mining-concessions dataset (government-
   sourced polygon boundaries, downloaded from raisg.org, clipped to this
   AOI in `data/leases/madre_de_dios_concessions.geojson`). Legal and
   illegal mining share the same excavated-bare-soil spectral signature --
   only the administrative status differs -- so active/titled concession
   interiors are a genuine positive-label source, and pixels >300m from
   any concession (active or not) with high NDVI are a genuine negative
   source. The RAISG "illegal mining" layer itself turned out too coarse
   to use directly (a single hand-drawn 2016 zone outline covering the
   whole region, ~240km x 140km, not per-pixel footprints).

   This also tightened the results substantially: "new disturbance" fell
   from ~8,500 ha/year (weak labels) to **~1,800 ha/year** (real labels),
   with alert polygons now typically 0.2-0.3 ha -- consistent with La
   Pampa's known pattern of fragmented small-scale artisanal mining sites,
   rather than the weak-label version's more diffuse over-classification.
   Algorithm f1 scores are still very high (0.997-1.000,
   `model_selection.py`'s comparison), but this is now a meaningfully
   different claim than before: the training task itself (obvious bare
   excavation vs. obvious dense forest) is genuinely easy to separate
   spectrally, so near-perfect accuracy on it is expected -- it does NOT
   mean the pipeline is 99.7% accurate on the actually-hard cases
   (ambiguous partially-cleared land, fallow, non-mining bare ground),
   which are deliberately excluded from training as unlabeled. Validating
   against those hard cases would need hand-labeled examples specifically
   targeting them.

4. **Hard-case validation -- measured.** Built a shared labeling tool
   (`data/hard_cases/`, published as an Artifact) to hand-label 40 chips
   stratified across genuinely ambiguous categories the classifier was never
   validated on: near-concession boundaries, river sandbars, its own
   50/50-probability cases, bare patches far from any known concession, and
   mid-range-NDVI vegetation. Results (`data/hard_cases/validation_summary.md`,
   single labeler, 2026-09-11 -- directional, not yet statistically
   conclusive): accuracy 0.625, precision 0.586, **recall 1.000** on the 32
   scored chips (8 marked "unsure" excluded). Recall being perfect matters
   for an alert system (it isn't missing real mining in this sample); the
   precision gap is concentrated almost entirely in two categories --
   `classifier_uncertain` (0.25 accuracy: at the model's own decision
   boundary it leans "mining", humans said "not mining" every time) and
   `ambiguous_vegetation` (0.17 accuracy: moderate vegetation loss from
   fallow land/agriculture/regrowth gets over-flagged as mining). Both point
   to the same cause already seen in the O3 seasonal-confound fix: the
   composite disturbance score leans too heavily on NDVI/NDMI (phenology-
   sensitive) relative to mining-specific BSI/SWIR signal.

5. **Hard-case validation -- fixed via hard_case_labels.py.** Confirmed the
   cause directly: the trained model's own feature importances rank NDVI
   (0.345) and NDWI (0.268) above BSI (0.192) and NDMI (0.098) -- the
   phenology-sensitive features dominate the mining-specific one, exactly
   as the accuracy pattern implied. Rather than hand-tuning a composite
   formula, the fix folds the real hand labels back into training directly
   (`hard_case_labels.py`): human labels for a small neighborhood (9x9px)
   around each hard-case chip's center override whatever concession-based
   label was there. Evaluated honestly with a stratified 50/50 split of the
   32 labeled chips (16 folded into training, 16 held out, never mixed --
   `data/hard_cases/run_augmented_eval.py`): F1 on the held-out 16 went
   **0.833 -> 0.909** (precision 0.714 -> 0.833, recall held at a perfect
   1.000, false positives 4 -> 2). `real_run.py` now folds in all 32 labeled
   hard cases for the deployed model (`data/hard_cases/votes.json` +
   `chips_data.json`), since the held-out evaluation already validated the
   approach -- standard practice: measure improvement on a genuine holdout,
   then retrain on the full labeled set for deployment.

   Sample size caveat still applies (32 chips, single labeler) -- next step
   for statistical confidence is more labelers on the same 40-chip set via
   the shared tool (votes are keyed per-labeler, so this needs no code
   changes, just more people using the link).

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the demo (no credentials needed)

Runs the full O1-O5 pipeline on a synthetic bi-temporal scene (one pre-existing
licensed mine, two new unlicensed excavation sites) so the architecture can be
demoed/validated before real Sentinel-2 data is wired in.

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m mining_detector.pipeline
```

Output: `outputs/alerts.geojson` — geo-referenced polygons with `confidence`
and `area_ha` per suspicious site.

## Run against real Sentinel-2 data

1. Register for free at https://dataspace.copernicus.eu
2. Set credentials:
   ```powershell
   $env:CDSE_USERNAME="you@example.com"
   $env:CDSE_PASSWORD="..."
   ```
3. Call `pipeline.run_live(bbox, date_t0, date_t1, lease_path)` for your pilot
   AOI. **Note:** the classifier in `detection.py` needs labeled training
   pixels for the real AOI before `run_live` can train it — this is currently
   a `NotImplementedError` stub pending either manual annotation of a few
   known mining/non-mining patches, or an open mining-footprint dataset.

## Faculty ML-methodology mapping

Two generic workflows were specified for the review: a tabular/CSV pipeline
(load -> preprocess -> split -> SMOTE -> train multiple algorithms -> check
overfit/underfit -> test -> select best model) and an image-file pipeline
(load -> preprocess -> CLAHE -> GAN augmentation -> split -> train -> test
-> select best model).

**CSV pipeline -> fully implemented, `model_selection.py`.** O2's per-pixel
spectral-feature table (NDVI/NDMI/BSI/NDWI + raw bands per pixel) *is* the
"CSV file" of that flowchart, just built in memory instead of read from
disk. Implemented exactly per the diagram:
- **Data split**: holdout (80:20 / 70:30 / 60:40, `run_holdout`) or
  stratified 10-fold CV (`run_cross_validation`), selectable via
  `MiningDetector.fit(..., cv_folds=10)`.
- **SMOTE**: applied to the training portion only (post-split / per-fold),
  since mining pixels are a minority class against background land cover.
- **Multiple algorithms**: Logistic Regression (L1 and L2), Random Forest,
  SVM (RBF kernel), Gradient Boosting -- trained and compared every run.
- **Overfitting/underfitting check + regularization**: train-vs-test F1 gap
  is computed per model and flagged past a threshold (`OVERFIT_GAP_THRESHOLD`
  in `model_selection.py`); L1/L2 penalty strength (`C`) covers the
  regularization axis for the linear models, and Gradient Boosting uses
  early stopping (`n_iter_no_change`) as its regularizer.
- **Selection of the best model**: highest test-set F1 across all
  candidates is auto-selected and used for O3-O5.

**Image/GAN pipeline -> planned future work, not implemented.** This track
(CLAHE contrast enhancement on raw tiles, then GAN/StyleGAN/CGAN-based
synthetic augmentation, then training a CNN/segmentation model) requires a
labeled image-patch dataset for illegal mining that does not yet exist for
this AOI. Building it (even via weak/self-supervised labels) and training a
GAN well is a multi-week effort on its own, which is why O2 currently ships
the spectral-classifier path above instead. Phase 2 plan: once ~50-100
hand-labeled or open-dataset mining/non-mining image patches exist for
Madre de Dios, apply CLAHE to normalize atmospheric haze common in Amazon
scenes, use a conditional GAN to synthesize additional positive (mining)
examples to correct the same class-imbalance problem SMOTE addresses in the
tabular track, then train a U-Net for pixel-level segmentation as a direct
upgrade path for O2 -- the feature-stack/probability-map interface in
`detection.py` is intentionally the same shape a CNN would output, so O3-O5
would not need to change.

## Why a tabular classifier, not a segmentation CNN?

O2 is framed in the project brief as "semantic segmentation," but no public
labeled dataset exists specifically for illegal-mining footprints. A
per-pixel classifier on spectral indices (NDVI/NDMI/BSI/NDWI + raw bands)
needs far fewer labeled pixels, trains in seconds, and stays explainable
(feature importances, per-algorithm metrics) — appropriate for a hackathon
timeline. `model_selection.py` compares five algorithms per run and picks
whichever scores best, rather than committing to one architecture upfront.
The code is structured so a U-Net can be swapped in later (same input
feature stack, same output probability map) without touching O3-O5.
