# Knee Fault V3 Decision

Status: experimental; rejected for production integration.

## Model

- Target: knees forward
- Method: V2 geometry/temporal features with inward-only hard negatives
- Selected hard-negative multiplier: 6.0
- Validation-selected threshold: 0.4763

## Held-Out Test Results

- Precision: 0.5686
- Recall: 0.7679
- F1: 0.6534
- Average precision: 0.6590
- ROC-AUC: 0.7706
- Forward-only recall: 0.7574
- False-forward ratio on inward-only frames: 0.6897
- False-forward ratio on neither frames: 0.3671

## Comparison With V2

- V2 inward-only false-forward ratio: 0.8793
- V3 inward-only false-forward ratio: 0.6897
- Improvement: 0.1896

V3 substantially reduced the primary cross-label confusion, but 68.97% of
inward-only frames are still incorrectly predicted as knees-forward.

Overall forward F1 declined from 0.6638 in V2 to 0.6534 in V3, and the
false-forward rate on neither frames remained essentially unchanged.

## Decision

Do not integrate V3 into production scoring.

Retain knees_forward_rf_v3.joblib as an experimental baseline. Build V4 as a
joint four-state classifier:

- neither
- forward_only
- inward_only
- both
