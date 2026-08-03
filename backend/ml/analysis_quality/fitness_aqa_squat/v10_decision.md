# Knee Analysis V10 Decision

## Experiment

V10 retained the V9 complete-rep detector, accepted repetitions,
ascent-based targets, and train/validation/test split.

Only the feature representation changed:

- V9: 709 phase-summary features
- V10: 761 fixed-length temporal features
- Setup: 4 interpolated points
- Descent: 8 interpolated points
- Bottom: 5 interpolated points
- Ascent: 8 interpolated points
- Finish: 4 interpolated points

V10 retained exactly:

- Train: 546 reps
- Validation: 115 reps
- Test: 122 reps

Target state counts matched V9 exactly.

## Forward-knee result

The unconstrained F1-selected threshold remains unsuitable because
false-forward warnings are excessive.

At the validation-constrained maximum FPR of 0.20, V10 improved over V9:

- Validation precision: 0.8600
- Validation recall: 0.5811
- Validation F1: 0.6935
- Validation FPR: 0.1707
- Validation specificity: 0.8293
- Validation balanced accuracy: 0.7052

Exploratory test at the validation-selected threshold:

- Precision: 0.7647
- Recall: 0.5417
- F1: 0.6341
- FPR: 0.2400
- Specificity: 0.7600
- Balanced accuracy: 0.6508

V10 is the leading forward-knee experiment, but it is not approved
for production deployment until it passes real-clip shadow testing.

## Inward-knee result

The inward model remains rejected.

Validation:

- Precision: 0.5000
- Recall: 0.2500
- F1: 0.3333

The target remains too sparse and performance is not strong enough
for user-facing coaching.

## Production decision

- Do not change production.
- Do not deploy the unconstrained threshold.
- Retain V10 for forward-knee shadow testing only.
- Do not deploy the inward model.
- Use the validation-constrained forward threshold with maximum
  validation FPR 0.20 for shadow evaluation.

## Real-clip shadow evaluation

V10 was evaluated on the same real clips used for V9.

### Knee-valgus control

Source:

- Back Squat- knee valgus_gold.mp4

Results:

- Pose rows: 196
- Detected reps: 2
- Conservative forward threshold: 0.525
- Rep 1 forward score: 0.2165
- Rep 2 forward score: 0.2194
- No false forward warnings

### Known forward-lean clip

Source:

- Back_squats_forward_lean_shadow.mp4

Default detector with minimum phase rows 4:

- Pose rows: 100
- Detected reps: 0

Shadow-only relaxed detector with minimum phase rows 2:

- Detected reps: 1
- Frames: 168 -> 180 -> 204
- Rows: 7
- Forward score: 0.3248
- Threshold: 0.525
- Decision: clear

The same relaxed repetition scored approximately 0.3776 under V9.
V10 therefore reduced the forward score on the known positive real clip.

## Final V10 decision

V10 is not approved for production or further threshold tuning.

Although fixed-length temporal resampling improved validation metrics and
the exploratory test metrics, it did not improve the decisive real-video
positive control.

The experiment demonstrates that offline validation improvement did not
transfer to this real clip.

Next work should focus on:

1. More reliable real-video rep boundaries.
2. Inspecting whether the detected seven-row repetition captures the
   actual forward-lean portion.
3. Better real-video training coverage or augmentation.
4. Avoiding threshold reduction based on a single positive clip.
