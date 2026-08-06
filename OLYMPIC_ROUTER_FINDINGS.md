# Olympic Router Findings

Release tested:
v20260805-release

Additional C&J validation:

1) v_CleanAndJerk_g01_c01.avi
- Expected: clean_and_jerk
- Got: split_jerk
- olympic_pred: clean_and_jerk
- olympic_conf: 0.748

2) v_CleanAndJerk_g02_c03.avi
- Expected: clean_and_jerk
- Got: clean
- olympic_pred: clean_and_jerk
- olympic_conf: 0.54

Conclusion:
Clean & jerk arbitration needs improvement.
The Olympic router detects C&J, but final routing sometimes collapses into subcomponents.

Future improvement:
- Add clean_and_jerk temporal priority when clean + jerk phases are both present.
- Avoid allowing split_jerk/clean labels to override a complete C&J sequence.
