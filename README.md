# Illegal Mining Detector

Sentinel-2-driven pipeline for detecting illegal/unregulated surface mining.
Maps to project objectives O1-O5:

| Stage | Objective | Module |
|---|---|---|
| O1 | Data acquisition (CDSE Sentinel-2 L2A) | `auth.py`, `search.py`, `download.py`, `preprocess.py` |
| O2 | Mining-area detection (spectral features + multi-algorithm classifier) | `indices.py`, `detection.py`, `model_selection.py`, `real_labels.py`, `hard_case_labels.py` |
| O3 | Bi-temporal change detection | `change_detection.py` |
| O4 | Legal / disputed / illegal footprint classification | `licensing.py` |
| O5 | Geo-referenced alert export (GeoJSON) | `alerts.py` |

`pipeline.py` is the single entry point for both modes:
- `pipeline.run_demo()` -- synthetic data, no credentials needed.
- `pipeline.run_live()` -- the real pipeline over the Peru pilot AOI (delegates to `real_run.py`).

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

6. **Terrain covariates -- implemented, negligible effect (honest negative
   result).** The SAR-based mining paper reviewed in Project Review 1
   identified persistent misclassification around peripheral/low-density
   mining structures and proposed adding elevation + hydrological-proximity
   covariates as a fix, since illegal mining correlates strongly with
   stream proximity. Implemented in `terrain.py`: real elevation from the
   Copernicus DEM GLO-30 (public AWS dataset, windowed COG reads via
   `/vsis3/`, no API key needed) and a distance-to-water raster derived
   from the scene's own NDWI. Wired into `indices.compute_feature_stack`
   and threaded through `detection.py`/`model_selection.py` as optional
   extra channels. Result on the real pilot AOI: both terrain features
   ranked **lowest of all 12 features** in the trained model (ELEVATION
   0.0004, DIST_WATER 0.0007, vs. NDWI at 0.316) -- negligible predictive
   value in this setup, most likely because NDWI (already in the feature
   stack) captures water-proximity information more directly than a
   derived distance metric, and this AOI's elevation is nearly flat
   (0-273m) with no strong mining/elevation relationship to exploit. This
   is a real, reproducible result, not a failed implementation -- the SAR
   paper's finding may hold in a different pilot region (e.g. hard-rock
   mining in mountainous terrain, where elevation gradients are much more
   informative).

7. **QC-SMOTE -- implemented, no measurable difference vs. standard SMOTE
   (honest negative result).** `qc_smote.py` is a documented
   reimplementation of Upman & Gowda (2026)'s quality-controlled SMOTE
   (not the authors' original code, which isn't publicly released --
   reconstructed from the paper's described mechanism: minority
   trustworthiness scoring, purity-checked synthesis, graceful degradation
   via duplication in noisy regions). Selectable via
   `smote_variant="qc"` in `detection.py`/`model_selection.py`. Tested
   against standard SMOTE on two real regimes: (a) the clean concession-
   based training set, and (b) the noisier hard-case-augmented held-out
   evaluation (Section 5 above) where real class-boundary ambiguity
   exists. **Both produced statistically identical results in both
   regimes.** Most likely explanation: Random Forest (the selected best
   model in both cases) aggregates many trees via majority voting, which
   buffers against exactly the synthetic-sample-placement differences
   QC-SMOTE targets -- a single decision tree or a placement-sensitive
   model (e.g. a small neural net) would be a fairer test of whether
   QC-SMOTE's mechanism matters for this specific pipeline.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Testing

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

44 tests covering every deterministic module (`indices.py`,
`change_detection.py`, `licensing.py`, `alerts.py`, `hard_case_labels.py`,
`model_selection.py`, `preprocess.py`, `weak_labels.py`, `real_labels.py`)
against synthetic data with known-correct answers -- no CDSE credentials or
downloaded imagery required.

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

Already working end-to-end for the Peru pilot AOI (Madre de Dios / La Pampa,
`config.py`):

1. Register for free at https://dataspace.copernicus.eu, then create an S3
   key pair at https://eodata-s3keysmanager.dataspace.copernicus.eu
   (band downloads use direct S3 access -- see "Why S3, not the OData
   product download?" below).
2. Copy `.env.example` to `.env` and fill in:
   ```
   CDSE_USERNAME=you@example.com
   CDSE_PASSWORD=...
   CDSE_S3_ACCESS_KEY=...
   CDSE_S3_SECRET_KEY=...
   ```
3. Download t0/t1 scenes for your AOI/dates (see `search.py` + `download.py`,
   or reuse the two scenes already cached under `data/raw/` for this pilot).
4. Run:
   ```powershell
   $env:PYTHONPATH="src"
   .\.venv\Scripts\python.exe -m mining_detector.real_run
   ```
   (equivalently `pipeline.run_live()`). Output: `outputs/real_alerts.geojson`,
   each polygon tagged `legal_status` (`disputed` or `illegal`), `confidence`,
   and `area_ha`.

To point this at a different AOI: update `config.PILOT_AOI_BBOX` and the
date constants, get a concessions/lease extract for that region in the same
shape as `data/leases/madre_de_dios_concessions.geojson` (a `situacion`
column distinguishing active from inactive statuses is optional but
improves O4's legal/disputed/illegal split -- see `real_labels.py`), and
either build a new `data/hard_cases/` labeled set for that AOI or skip that
step (`real_run.py` folds it in only when present).

### Why S3, not the OData product download?

CDSE's OData `$value` endpoint downloads the entire SAFE product archive
(800MB-1GB) and repeatedly truncated mid-stream in testing (`download.py`
still has a retry/resume-capable `download_product_zip` as a documented
fallback). Direct S3 access to the `eodata` bucket only pulls the ~6 bands
this project needs (tens of MB total) and was reliable in testing.

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

## 5 hybrid deep-learning architectures (proof-of-concept scale)

`src/mining_detector/hybrid_models/` implements five hybrid architectures
requested for comparison: YOLO+ViT+LSTM, U-Net+Transformer, CNN+BiLSTM,
Siamese CNN+Transformer, and ResNet+Attention U-Net. Read this section
before citing the results below -- what they demonstrate is narrower than
it might first appear.

**What's real**: every architecture is a genuine PyTorch implementation of
its named components (real conv backbones, real `nn.TransformerEncoderLayer`
multi-head attention, real LSTM/BiLSTM, a real shared-weight Siamese
encoder, real residual blocks, a real attention-gate mechanism per the
Attention U-Net paper) -- trained with real forward/backward passes, not
mocked. The Siamese model uses genuine bi-temporal input: paired t0/t1
crops at the same pixel location, pixel-aligned (verified: identical
`(6, 2776, 3264)` shape and affine transform between scenes).

**What's *not* real segmentation/detection**: YOLO and U-Net were designed
for bounding-box and pixel-mask outputs respectively. This project has
neither -- only a single mining/not-mining label per chip (from the
hard-case labeling tool, see `data/hard_cases/`) -- so every architecture
here is adapted with a classification head. For the two segmentation-style
architectures (U-Net+Transformer, ResNet+Attention U-Net), the decoder is
architecturally intact (real upsampling, real skip connections, a real
attention gate) but is global-average-pooled into a single classification
logit rather than supervised against pixel masks, since none exist.

**What the sample size means**: training data is the 32 usable labels
(mining/not-mining, "unsure" excluded) from the 40-chip hard-case set --
24 for training, 8 held out for testing, per `dataset.py`'s stratified
split. Every model reaches 100% training accuracy and ~87.5% test accuracy
(7/8 correct) after 30 epochs. **100% training accuracy with ~20-40k
parameters on 24 examples is textbook overfitting, not a sign these are
good models** -- with a test set of 8, "87.5% accuracy" means exactly
7 or 8 correct, nothing more granular is statistically meaningful. Treat
these results as confirmation that each architecture is correctly wired
end-to-end (real gradients flow, real predictions come out), not as
validated performance claims -- that would require the labeled image
dataset (segmentation masks, hundreds+ of examples) this project doesn't
have. See `data/hybrid_models_results/hybrid_results.json` for full
per-model metrics.

To run: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`,
then `python -m mining_detector.hybrid_models.train` (PYTHONPATH=src).

### Upgrade: real segmentation on a real labeled dataset

The proof-of-concept caveat above is now only true for `siamese_cnn_transformer`.
The other four were retrained on a real, independently-sourced labeled
dataset (Open-Pit Mining Areas Dataset -- 10 512x512 satellite image/binary-
mask TIFF pairs; not redistributed via git, drop it at
`data/external/open_pit_mining_dataset/Data/` to reproduce; no license file
was included with the download, so cite wherever you originally obtained it
before submitting). Tiled into 64x64 patches (512 train / 128 test,
held out from 2 of the 10 source tiles entirely so no train patch is
spatially adjacent to a test patch) via `openpit_dataset.py`:

| Model | Task | Test result | n (test) |
|---|---|---|---|
| `unet_transformer` | **real per-pixel segmentation** | IoU 0.541, Dice 0.577 | 128 patches |
| `resnet_attention_unet` | **real per-pixel segmentation** | IoU 0.254, Dice 0.320 | 128 patches |
| `yolo_vit_lstm` | classification (real patch labels) | acc 0.945, F1 0.863 | 128 patches |
| `cnn_bilstm` | classification (real patch labels) | acc 0.844, F1 0.655 | 128 patches |
| `siamese_cnn_transformer` | classification (32-chip set, unchanged) | acc 0.875, F1 0.889 | 8 chips |

Run: `python -m mining_detector.hybrid_models.train_openpit` (PYTHONPATH=src).
Results: `data/hybrid_models_results/openpit_results.json`.

**Read this before citing these numbers**: the held-out test set comes from
2 tiles drawn from only 2 underlying source scenes (the other 8 training
tiles are also from those same 2 scenes) -- so this measures interpolation
within two known mining sites, not generalization to a genuinely new
location. `resnet_attention_unet` underperforming `unet_transformer` on
segmentation is a real, reproducible result in our tests, not a bug --
plausibly because its decoder produces a coarser native resolution (32x32,
upsampled to match) versus `unet_transformer`'s native 64x64 output, though
we have not isolated the exact cause. `siamese_cnn_transformer` is
intentionally NOT retrained on this dataset: it needs genuine bi-temporal
pairs, which this single-date dataset doesn't have, so it correctly stays
on the 32 real paired t0/t1 hard-case chips instead of being forced onto
data that doesn't match its design.

### Lightweight-variant benchmark (compute-efficiency future work)

MineFormer and TransSiamUNet (both cited in Project Review 1) name
"further reduction of computational cost... for real-time or edge-device
deployment" as future work. `benchmark_lite.py` builds a lightweight
variant of each architecture (halved embedding width, single
attention/LSTM layer where applicable -- see `LITE_KWARGS` in `models.py`)
and trains/evaluates both scales on the exact same real data each
architecture already uses:

| Model | Params (full → lite) | Latency (full → lite) | Result (full → lite) |
|---|---|---|---|
| `unet_transformer` | 29,523 → 5,499 | 3.25ms → 0.96ms | IoU 0.582 → 0.285 (full clearly better) |
| `resnet_attention_unet` | 35,532 → 9,128 | 1.27ms → 0.91ms | IoU 0.274 → **0.475 (lite better)** |
| `yolo_vit_lstm` | 40,065 → 8,209 | 1.18ms → 0.79ms | acc 0.914 → 0.914, F1 0.807 → 0.792 (near-identical, 5x cheaper) |
| `cnn_bilstm` | 22,145 → 7,217 | 0.66ms → 0.58ms | F1 **0.000 → 0.605 (lite far better)** |
| `siamese_cnn_transformer` | 22,305 → 3,681 | 0.92ms → 0.71ms | acc/F1 identical, 6x cheaper |

Run: `python -m mining_detector.hybrid_models.benchmark_lite` (PYTHONPATH=src).
Results: `data/hybrid_models_results/lite_benchmark_results.json`.

**This is not a uniform "smaller is worse" story, and that's the actual
finding.** Two of five lite variants matched the full model at a fraction
of the cost (Siamese, YOLO+ViT+LSTM); two lite variants **outperformed**
their full counterpart (ResNet+Attention U-Net, CNN+BiLSTM -- the full
CNN+BiLSTM collapsed to predicting only the majority class, F1=0.000
despite 80% accuracy, a classic imbalance failure that the smaller model
did not exhibit); only `unet_transformer` showed the expected
capacity-accuracy tradeoff. This directly echoes the "When Less Is More"
paper's own thesis (also cited in Project Review 1): on a small training
set, a larger model's extra capacity does not reliably translate to
better generalization, and can actively hurt it via overfitting.
