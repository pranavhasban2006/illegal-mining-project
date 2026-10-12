# Engine notes (for the README)

Owner: Terminal A. Terminal B copies or summarises this into the README.

## What the engine does

1. Finds the least cloudy Sentinel-2 L2A scene within 15 days of each date. It uses the AWS
   Earth Search catalogue and falls back to Microsoft Planetary Computer.
2. Reads only the window that covers the area, resamples to a 10 m UTM grid, and masks clouds,
   shadows and cirrus with the SCL band. Both dates share one grid. Results are cached in
   `data/cache/`.
3. Builds labels from public data only: Global Mining Footprint polygons (positives) and
   dense vegetation or permanent water (negatives). Hand labels can override them.
4. Trains five candidate classifiers (L1 and L2 logistic regression, random forest, RBF SVM,
   gradient boosting) on a spatial train/test split, with SMOTE applied to the training part only.
   The best test F1 wins. Overfitting is flagged when the train-test F1 gap exceeds 0.05.
5. Normalises each scene on its own before prediction, so haze and season do not look like change.
6. Flags new disturbance where P(mining, t1) is high and P(mining, t0) is low, then cleans the
   mask, drops patches under 0.5 ha and labels each patch legal, disputed or illegal.

## Run it

```
python -m mining_detector demo                 # offline, synthetic, a few seconds
python -m mining_detector fetch --date 2025-07-27   # check that satellite access works
python -m mining_detector live --example       # real data, default example area
python -m mining_detector live --bbox -72.10 -14.45 -71.95 -14.30 --t1 2025-07-27
```

## Known limitations

- **Weak labels.** Mining footprints are a public, coarse dataset. Coverage is patchy: the
  example area has 13 usable polygons, and the La Pampa box has none. Expect label noise.
- **No accuracy claim on new regions.** Scores come from a spatial hold-out inside one area.
  They say nothing about other terrain.
- **Model selection uses the same hold-out.** The reported F1 is slightly optimistic.
- **Ambiguous land is not handled.** Roads, quarries, farms and burn scars can look like mining.
- **Small patches can be missed** because of cloud masking and the 0.5 ha minimum.
- **Alerts are not legal findings.** Check legal status against official concession records.
- **Demo output is synthetic.** It shows that the pipeline works, not a real detection.
