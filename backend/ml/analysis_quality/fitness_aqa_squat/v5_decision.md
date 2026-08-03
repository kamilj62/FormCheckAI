# Knee Fault V5 Decision

Status: experimental; rejected for production integration.

## Model

Joint four-state Random Forest using V5 robust geometry:

- unstable width divisions removed
- torso-normalized width differences
- left/right knee-line offsets
- heel-width features
- explicit frontal/side camera confidence
- temporal deltas

## Held-Out Test Results

- Balanced accuracy: 0.4667
- Macro F1: 0.4434

### Derived Knees Forward

- Precision: 0.5714
- Recall: 0.7358
- F1: 0.6433

### Derived Knees Inward

- Precision: 0.3128
- Recall: 0.5182
- F1: 0.3901

### Critical Joint-State Results

- False-forward on inward-only: 0.6207
- Correct inward-only state: 0.3448
- False-forward on neither: 0.3497
- Correct forward-only state: 0.6056
- Correct both state: 0.2785

## Comparison With V4

- Macro F1: 0.4393 -> 0.4434
- Forward F1: 0.6456 -> 0.6433
- Inward F1: 0.3536 -> 0.3901
- Inward recall: 0.4453 -> 0.5182
- False-forward on inward-only: 0.6207 -> 0.6207
- False-forward on neither: 0.3609 -> 0.3497

## Decision

V5 improves knees-inward recall and F1 and removes unstable geometry ratios.
However, it does not improve the primary inward-only versus forward confusion.

Do not integrate V5 into production scoring.

Retain knee_joint_robust_geometry_rf_v5.joblib as an experimental baseline.
The next revision should move from frame-level classification to rep-level or
interval-level temporal summaries.
