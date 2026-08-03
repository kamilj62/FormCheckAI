# Knee Fault V4 Decision

Status: experimental; rejected for production integration.

## Model

Joint four-state Random Forest using V2 geometry and temporal features:

- neither
- forward_only
- inward_only
- both

## Held-Out Test Results

- Balanced accuracy: 0.4572
- Macro F1: 0.4393

### Derived Knees Forward

- Precision: 0.5683
- Recall: 0.7474
- F1: 0.6456

### Derived Knees Inward

- Precision: 0.2933
- Recall: 0.4453
- F1: 0.3536

### Critical Joint-State Results

- False-forward on inward-only: 0.6207
- Correct inward-only state: 0.3276
- False-forward on neither: 0.3609
- Correct forward-only state: 0.6221
- Correct both state: 0.2532

## Comparison With V3

- V3 false-forward on inward-only: 0.6897
- V4 false-forward on inward-only: 0.6207
- Improvement: 0.0690

- V3 forward F1: 0.6534
- V4 forward F1: 0.6456

## Decision

V4 modestly reduced the primary inward-only versus forward confusion, but
62.07% of inward-only frames are still predicted as knees-forward.

The joint model also performs poorly on the rare both class and slightly
reduces derived knees-forward F1 compared with V3.

Do not integrate V4 into production scoring.

Retain knee_joint_four_state_rf_v4.joblib as an experimental baseline.
The next revision should improve the feature representation rather than only
changing class weighting or classifier structure.
