# Knee Fault V2 Decision

Status: experimental; rejected for production integration.

## Knees Forward Test

- Precision: 0.5710
- Recall: 0.7927
- F1: 0.6638
- Average precision: 0.6689
- ROC-AUC: 0.7747

## Knees Inward Test

- Precision: 0.2845
- Recall: 0.4818
- F1: 0.3577
- Average precision: 0.2998
- ROC-AUC: 0.8636

## Critical Confusion

- False-forward ratio on inward-only frames: 0.8793
- Correct-inward ratio on inward-only frames: 0.5862
- False-forward ratio on neither frames: 0.3650
- False-inward ratio on neither frames: 0.0613

## Decision

V2 substantially improves both fault models over V1, especially knees inward.
However, the knees-forward model still confuses inward-only frames with
knees-forward 87.93% of the time and fires on 36.5% of neither frames.

Do not integrate either model into production scoring yet.
Retain both models as experimental baselines for V3.
