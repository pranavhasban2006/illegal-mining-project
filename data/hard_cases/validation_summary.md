# Hard-case validation results

Single-labeler pilot (Prakul Jain), 40 chips, 2026-09-11. Small sample /
single rater -- directional, not statistically conclusive -- but the first
real out-of-sample validation the classifier has had (see README's
weak-label-ceiling discussion for why this was needed).

## Overall (32 scored, 8 "unsure" excluded)

| Metric | Value |
|---|---|
| Accuracy | 0.625 |
| Precision | 0.586 |
| Recall | 1.000 |
| F1 | 0.739 |

Confusion: TP=17 FP=12 TN=3 FN=0

## Per category

| Category | n | Accuracy | Note |
|---|---|---|---|
| river_sandbar | 7 | 1.00 | Sandbars here are largely real mining/dredging activity |
| far_bare_random | 5 | 1.00 | Correctly flags undiscovered bare patches |
| boundary_zone | 6 | 0.83 | Solid near concession edges |
| classifier_uncertain | 8 | 0.25 | At the model's own 0.5 boundary, it leans "mining"; humans said "not mining" every time |
| ambiguous_vegetation | 6 | 0.17 | Model over-flags moderate vegetation loss as mining |

## Takeaway

Recall is perfect (never misses real mining in this sample) but precision
fails specifically on land with reduced vegetation that isn't bare soil --
fallow land, agriculture, secondary regrowth. The composite disturbance
score leans too heavily on NDVI/NDMI (phenology-sensitive, as already seen
in the O3 seasonal-confound issue) relative to mining-specific bare-soil
signatures (BSI/SWIR). Concrete next step: re-weight the feature importance
away from NDVI/NDMI toward BSI/SWIR in indices.py's composite score, or
raise the decision threshold for the "far from concession" regime
specifically, then re-validate against this same 40-chip set to confirm
the fix.

Next step for statistical confidence: get 2-3 more people to label the
same 40 chips (the tool supports it -- votes are keyed per labeler) to
measure inter-rater agreement before trusting single-rater numbers fully.
