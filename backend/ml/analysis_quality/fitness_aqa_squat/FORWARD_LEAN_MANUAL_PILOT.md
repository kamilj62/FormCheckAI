# Manual Excessive Forward Lean Pilot

## Status

Shadow-only feasibility experiment.

No production analysis, scoring, routing, or coaching behavior was changed.

The test split was not used.

## Ground truth

A manual review target was created with four labels:

- `clear`
- `excessive_forward_lean`
- `uncertain`
- `unusable`

Initial annotations were made from five-phase contact sheets:

- setup
- descent
- bottom
- ascent
- finish

Selected validation disagreements were subsequently reviewed in motion.

## Annotation totals

### Training

- Total reviewed: 60
- Clear: 15
- Excessive forward lean: 19
- Uncertain: 7
- Unusable: 19
- Usable binary rows: 34

### Validation

- Total reviewed: 20
- Clear: 6
- Excessive forward lean: 9
- Uncertain: 2
- Unusable: 3
- Usable binary rows: 15

## Motion review

Candidates 61, 63, and 64 were reviewed in motion.

- Candidate 61 changed from provisional `excessive_forward_lean` to `clear`.
- Candidate 63 remained `clear`.
- Candidate 64 remained `clear`.
- All three were upgraded to high-confidence annotations.

The motion review showed that torso inclination alone is not sufficient. A valid assessment must distinguish natural torso inclination from dynamic chest collapse or hips rising substantially faster than the shoulders.

## Camera filtering

The existing derived fields:

- `frontal_view_confidence`
- `side_view_confidence`

did not reliably separate usable sagittal views from unusable front or rear views.

A camera filter was derived exclusively from reviewed training candidates:

- median projected shoulder width `< 0.60`
- median projected hip width `< 0.40`

Observed training-review behavior:

- usable recall: 0.9118
- unusable rejection: 0.6316

The filter was applied to validation candidate selection without using validation labels.

## Full V9 ExtraTrees baseline

Dataset:

- Training rows: 34
- Validation rows: 15
- Feature count: 709
- Test split used: no

Model configuration:

- ExtraTreesClassifier
- 500 estimators
- maximum depth 4
- minimum leaf size 2
- square-root feature sampling
- balanced class weights
- threshold 0.50

Results after motion-review corrections:

- Train balanced accuracy: 1.0000
- Validation balanced accuracy: 0.8333
- Validation ROC-AUC: 0.8889
- Validation confusion matrix: `[[4, 2], [0, 9]]`
- Excessive-forward-lean precision: 0.8182
- Excessive-forward-lean recall: 1.0000

Known false positives:

- Candidate 63
- Candidate 64

Both were reviewed in motion and confirmed clear.

## Compact 24-feature experiment

A compact model used selected torso-angle, hip-motion, shoulder-motion, hip-angle, and knee-angle features.

Results:

- Train balanced accuracy: 0.9737
- Validation balanced accuracy: 0.5556
- Validation ROC-AUC: 0.6852
- Validation confusion matrix: `[[4, 2], [5, 4]]`
- Precision: 0.6667
- Recall: 0.4444

Decision:

The compact model was rejected. It retained the same two false positives and introduced five false negatives.

## Conclusion

The manually labeled target appears learnable, and the full V9 representation produced a promising feasibility result.

It is not deployment-ready because:

- only 34 usable training rows exist
- only 15 usable validation rows exist
- many annotations originated from static contact sheets
- training performance remains perfect, indicating overfitting risk
- validation specificity remains limited
- no untouched test evaluation was performed

The next recommended step is to collect and motion-review additional side-view clips, especially clear negatives resembling Candidates 63 and 64.

Do not tune further thresholds or feature subsets against the current 15-row validation set.
