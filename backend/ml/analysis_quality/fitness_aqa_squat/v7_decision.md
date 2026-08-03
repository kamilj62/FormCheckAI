# Knee Fault V7 Decision

Status: experimental; rejected for production integration.

## Design

V7 was a geometry-focused ablation of V6.

It removed:

- all slope features
- all delta features
- segment duration
- segment frame count

It retained 338 percentile, range, bottom-phase, geometry, visibility, and
camera-related features.

## Validation Results

### Forward

- Threshold: 0.365
- Precision: 0.4901
- Recall: 0.8810
- F1: 0.6298
- ROC-AUC: 0.6871
- Average precision: 0.5701
- R2: 0.0396

### Inward

- Threshold: 0.33
- Precision: 0.3333
- Recall: 0.2941
- F1: 0.3125
- ROC-AUC: 0.6782
- Average precision: 0.2152
- R2: -0.0976

## Exploratory Test Results

### Forward

- Precision: 0.4630
- Recall: 0.8333
- F1: 0.5952
- ROC-AUC: 0.6381
- Average precision: 0.5816
- False positives: 87
- True negatives: 35

### Inward

- Precision: 0.2353
- Recall: 0.2105
- F1: 0.2222
- ROC-AUC: 0.7982
- Average precision: 0.3144

### Critical Joint-State Results

- False-forward on neither: 0.7043
- False-forward on inward-only: 0.8571
- Correct inward-only state: 0.1429
- Correct forward-only state: 0.6795
- Correct both state: 0.1667

## Comparison With V6

Removing temporal slope and delta features reduced both forward and inward
performance and increased the false-forward rate on neither intervals.

The temporal features contained useful signal, but V6 encoded them without
normalizing descent and ascent direction.

## Decision

Do not integrate V7 into production.

Do not continue with geometry-only ablations.

The next experiment should retain temporal information but express it using
phase-normalized setup-to-bottom and bottom-to-finish measurements.
