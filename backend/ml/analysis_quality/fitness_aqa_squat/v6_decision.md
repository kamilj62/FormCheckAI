# Knee Fault V6 Decision

Status: experimental; rejected for production integration.

## Design

V6 replaced frame-level four-state classification with interval-level modeling.

Two Extra Trees regressors independently predicted:

- forward_fraction
- inward_fraction

Each interval was summarized using 392 temporal and geometry features.
Decision thresholds were selected using validation data only.

## Dataset

- Train intervals: 954
- Validation intervals: 208
- Test intervals: 212

Majority-positive counts:

- Forward: train 414, validation 84, test 90
- Inward: train 80, validation 17, test 19

Joint inward-only counts were extremely small:

- Validation: 4
- Test: 7

## Locked Test Results

### Forward

- MAE: 0.2984
- RMSE: 0.3449
- R2: 0.1863
- Threshold: 0.335
- Precision: 0.5417
- Recall: 0.8667
- F1: 0.6667
- ROC-AUC: 0.7319
- Average precision: 0.6181

Confusion matrix:

- True negatives: 56
- False positives: 66
- False negatives: 12
- True positives: 78

### Inward

- MAE: 0.1461
- RMSE: 0.2132
- R2: 0.0815
- Threshold: 0.32
- Precision: 0.4000
- Recall: 0.4211
- F1: 0.4103
- ROC-AUC: 0.8514
- Average precision: 0.3929

Confusion matrix:

- True negatives: 181
- False positives: 12
- False negatives: 11
- True positives: 8

### Critical Joint-State Results

- False-forward on inward-only: 0.8571
- False-forward on neither: 0.5217
- Correct inward-only state: 0.1429
- Correct forward-only state: 0.7308
- Correct both state: 0.5000

## Decision

Do not integrate V6 into production.

Interval aggregation did not resolve the primary confusion between knees-forward
and knees-inward. The forward threshold produced excessive false positives, and
the validation/test sets contain too few inward-only intervals for stable model
selection.

The next step should be a dataset and split audit rather than another model
variant. In particular:

1. Check whether filename-prefix groups overlap between train, validation, and
   test.
2. Inspect whether the same person, recording session, or source appears across
   splits.
3. Review knees-forward and knees-inward interval boundaries for annotation
   consistency.
4. Build a new group-separated evaluation split before further model tuning.
