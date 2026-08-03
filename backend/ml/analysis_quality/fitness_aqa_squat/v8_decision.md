# Knee Fault V8 Decision

Status: experimental; partial success; not approved for production.

## Design

V8 replaced raw interval-wide slopes with phase-normalized features:

- setup
- descent
- bottom
- ascent
- finish
- setup-to-bottom change
- bottom-to-finish change
- setup-to-finish change
- phase-specific ranges
- setup-relative peak deviations

Because more than half of annotation intervals placed the detected bottom at an
endpoint, V8 classified intervals as:

- complete
- descent-only
- ascent-only
- short or ambiguous

Unavailable phase features were masked and explicit phase-availability
indicators were added.

## Dataset

- Train: 954 intervals, 716 features
- Validation: 208 intervals, 716 features
- Test: 212 intervals, 716 features

Complete candidates represented only about 34-36 percent of intervals.

## Validation

### Forward

- Threshold: 0.38
- Precision: 0.5333
- Recall: 0.8571
- F1: 0.6575
- ROC-AUC: 0.7337
- Average precision: 0.6088
- R2: 0.1512

### Inward

- Threshold: 0.225
- Precision: 0.2000
- Recall: 0.5882
- F1: 0.2985
- ROC-AUC: 0.7481
- Average precision: 0.1929
- R2: -0.1090

## Exploratory Test

### Forward

- Precision: 0.5704
- Recall: 0.8556
- F1: 0.6844
- ROC-AUC: 0.7428
- Average precision: 0.6019
- False-forward on neither: 0.4522
- False-forward on inward-only: 0.8571

Compared with V6:

- Forward F1 improved from 0.6667 to 0.6844.
- Forward ROC-AUC improved from 0.7319 to 0.7428.
- False-forward on neither improved from 0.5217 to 0.4522.
- False-forward on inward-only did not improve.

### Inward

- Precision: 0.2619
- Recall: 0.5789
- F1: 0.3607
- ROC-AUC: 0.8549
- Average precision: 0.3284

V6 inward F1 remained better at 0.4103.

## Interval-Type Threshold Experiment

Validation-derived forward thresholds:

- complete: 0.38
- descent-only: 0.265
- ascent-only: 0.385
- short or ambiguous: 0.475

On exploratory test data, type-specific thresholds reduced forward F1 from
0.6844 to 0.6814 and increased false-forward on neither from 0.4522 to 0.4609.

Type-specific thresholds are rejected.

## Decision

- Retain V8 as the leading experimental forward-knee model.
- Retain V6 as the stronger experimental inward-knee model.
- Do not integrate either model into production.
- Do not continue threshold tuning on the repeatedly inspected test split.
- Build the next dataset around complete squat repetitions rather than
  fragmented annotation intervals.
