# Back Squat Analysis V2 — Current Baseline

This document records the behavior of `analyze_squat_reps` before any
Back Squat Analysis V2 threshold changes.

## Rep detection

For non-overhead squats:

- Rep-entry threshold:
  `35th percentile of knee angle`
- A rep begins when knee angle falls below the threshold.

## Measurements

Measurements are calculated per detected rep.

| Measurement | Calculation |
|---|---|
| Depth | 20th percentile of clipped knee angle |
| Torso | 75th percentile of clipped torso angle |
| Knee tracking | 25th percentile of clipped valgus ratio |
| Heel lift | 90th percentile of clipped heel-lift signal |
| Neck drop | 85th percentile of head-drop signal |
| Neck forward | 85th percentile of head-forward signal |

## Back squat thresholds

### Depth

| Grade | Rule |
|---|---|
| Good | knee angle <= 115 |
| Borderline | knee angle <= 130 |
| Poor | knee angle > 130 |

### Torso

| Grade | Rule |
|---|---|
| Good | torso angle <= 60 |
| Borderline | torso angle <= 75 |
| Poor | torso angle > 75 |

### Knee tracking

| Grade | Rule |
|---|---|
| Good | valgus ratio >= 0.98 |
| Borderline | valgus ratio >= 0.85 and < 0.98 |
| Poor | valgus ratio < 0.85 |

### Heel lift

| Grade | Rule |
|---|---|
| Good | heel lift <= 0.03 |
| Borderline | heel lift > 0.03 and <= 0.045 |
| Poor | heel lift > 0.045 |

### Neck position

Poor when either:

- neck drop > 0.14
- neck forward > 0.18

Borderline when either:

- neck drop > 0.09
- neck forward > 0.13

Otherwise good.

## Score penalties

Each rep begins at 10.0.

| Category | Borderline | Poor |
|---|---:|---:|
| Depth | 0.6 | 1.4 |
| Torso | 1.0 | 2.2 |
| Knees | 1.5 | 3.5 |
| Heels | 0.4 | 0.9 |
| Neck | 0.8 | 1.8 |

Final score is clamped to a minimum of 1.0 and rounded to one decimal.

## Known limitations

- Thresholds are not calibrated against manually labeled coaching faults.
- Knee valgus uses a 2D ratio and may vary substantially by camera angle.
- Depth is inferred from knee angle rather than hip crease relative to knee.
- Heel lift uses normalized pose coordinates rather than direct foot contact.
- Torso angle thresholds do not account for anatomy or squat style.
- Fault precision and recall are currently unknown.
