# Knee Fault V9 Decision

Status: experimental; complete-rep forward detector is promising but not
approved for production.

## Dataset design

V9 operates on continuous raw pose sequences rather than V5-filtered frames.

Complete squat repetitions are detected using:

- upright setup before the bottom
- upright finish after the bottom
- minimum descent and ascent sample counts
- minimum source-video span
- knee-angle range
- hip-drop validation

Detected repetitions:

- Train: 546
- Validation: 115
- Test: 122

Each repetition contains 709 phase-normalized geometry features across:

- setup
- descent
- bottom
- ascent
- finish

The training targets are soft annotation fractions during ascent:

- ascent_forward_fraction
- ascent_inward_fraction

At the 0.5 reporting threshold, the training set contains:

- Forward positive: 307
- Inward positive: 58

## Original validation-selected thresholds

### Forward threshold 0.365

Validation:

- Precision: 0.6857
- Recall: 0.9730
- F1: 0.8045
- ROC-AUC: 0.7268
- False-forward on neither: 0.8000

Exploratory test:

- Precision: 0.6075
- Recall: 0.9028
- F1: 0.7263
- ROC-AUC: 0.6817
- False-forward on neither: 0.8222

This threshold is rejected because it flags nearly every clean repetition.

### Inward threshold 0.14

Validation:

- Precision: 0.2000
- Recall: 0.5833
- F1: 0.2979
- ROC-AUC: 0.6837

Exploratory test:

- Precision: 0.2105
- Recall: 0.4706
- F1: 0.2909
- ROC-AUC: 0.6650

The inward detector is rejected.

## Specificity-constrained forward threshold

A validation-derived maximum false-positive-rate constraint was evaluated.

Selected conservative threshold: 0.52

Validation:

- Precision: 0.8367
- Recall: 0.5541
- F1: 0.6667
- False-positive rate: 0.1951
- Specificity: 0.8049
- Balanced accuracy: 0.6795

Exploratory test:

- Precision: 0.7400
- Recall: 0.5139
- F1: 0.6066
- False-positive rate: 0.2600
- Specificity: 0.7400
- Balanced accuracy: 0.6269

Threshold 0.52 is retained as the conservative experimental forward threshold.

## Decision

- Retain V9 complete-rep extraction and feature generation.
- Retain the forward model with threshold 0.52 for shadow-mode testing.
- Reject the original F1-selected threshold 0.365.
- Reject the V9 inward model.
- Do not integrate V9 into production-facing coaching yet.
- Evaluate V9 on real gym clips and manually inspect false positives and false
  negatives before another model revision.

## Real-clip shadow evaluation

### Knee-valgus control

Video:

`Back Squat- knee valgus_gold.mp4`

Results:

- Pose rows: 196
- Complete reps detected: 2
- Forward scores: 0.2405 and 0.2766
- Forward warnings at threshold 0.52: 0

V9 correctly avoided confusing the detected valgus repetitions with a
forward-knee fault. However, production analysis previously found nine
repetitions, so V9 rep coverage was poor.

### Forward-lean positive control

Video:

`Back squats- forward lean.mov`

The original MOV had decoding/indexing problems and was converted to a
constant-frame-rate MP4.

Results with the training detector settings:

- Pose rows: 100
- Complete reps detected: 0
- Main rejection: bottom_not_deep_enough

Curve inspection showed that a valid squat occurred around frames 168-204,
but the 5 FPS extraction provided only two descent rows and four ascent rows.

A shadow-only minimum-phase override of two rows detected one repetition:

- Frames: 168 -> 180 -> 204
- Sampled rows: 7
- Descent rows: 2
- Ascent rows: 4
- Forward score: 0.3776
- Forward warning at threshold 0.52: no

This is a false negative on a known forward-lean example. It is also outside
the V9 training distribution, which required at least four sampled rows in
each phase and at least nine rows per repetition.

## Final V9 shadow decision

- Do not lower the forward threshold below 0.52.
- Do not deploy the V9 forward model.
- Do not deploy the V9 inward model.
- Do not replace production rep detection with the V9 detector.
- Retain V9 as research evidence that complete-rep context is useful, but the
  current 5 FPS sampling and variable-length phase representation are not
  robust to fast real-world repetitions.
- A future model should use production rep boundaries and fixed-length
  phase resampling rather than minimum sampled-row requirements.
