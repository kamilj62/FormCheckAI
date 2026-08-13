# Router Cleanup Plan

This plan freezes one-off classifier patches and moves the app toward one
family-first router.

## Current Production Flow

`backend/app/main.py` is still the production router. In broad order, one
analysis currently does this:

1. Extract pose and biomechanics.
2. Run the base movement model.
3. Run the biomechanics classifier.
4. Run the squat variant router.
5. Run Olympic gating plus the active Olympic router.
6. Run the bodyweight router.
7. Build Router V6 audit scores.
8. Run movement protections.
9. Apply many inline final-label corrections in `main.py`.
10. Run Router V5 Olympic corrections.
11. Build Router V8 diagnostics, with one production C&J lock exception.
12. Run central/family/press/hierarchical/specialist shadow routers.
13. Let `simplify_final_classification` optionally change production output.
14. Return the final response.

That is too many layers. The same decision can be made or changed in several
places, which makes failures hard to reason about.

## Keep

These should remain as core inputs to the final router:

- `movement_signatures.py`: canonical exercise/family map.
- Base movement model: broad evidence.
- Biomechanics classifier: broad evidence.
- Squat variant router: squat specialist evidence.
- Bodyweight router: bodyweight specialist evidence.
- Olympic router plus Router V5 events: Olympic specialist evidence.
- `router_audit.py`: debug scoring only.
- `run_analyzer_audit.py`: benchmark scoreboard.

## Merge

These should become one production decision module:

- `router_v8/protections.py`
- `final_classifier.py`
- `final_press_recovery.py`
- `final_bench_recovery.py`
- `squat_variant_recovery.py`
- the remaining inline final-label correction block in `main.py`

Target module:

- `backend/app/ml/final_decision_router.py`

Target shape:

1. Build a `RouterContext` from all model/signal outputs.
2. Pick a family: `press`, `bodyweight`, `olympic`, `squat`, `hinge`.
3. Pick the best label inside that family.
4. Apply a small conflict resolver.
5. Return one decision object with label, confidence, mode, and reason.

## Audit-Only

These should stop changing production labels until they beat the current router
on the 18-exercise audit:

- `central_router_shadow.py`
- `family_router_shadow.py`
- `hierarchical_router_shadow.py`
- `press_variant_shadow.py`
- `specialist_router_stack.py`
- Router V8 fusion output except for any explicitly migrated production rule.

## Delete Candidates

Delete after their useful logic is migrated or proven unnecessary:

- Most of `router_v8/locks.py`: it is large and mostly a patch stack.
- The late final recovery rules in `main.py` that only explain one clip.
- Any shadow router that only duplicates the family-first plan.
- Generated or obsolete router version files already removed from
  `router_v8/`.

## Stop Rules

Do not add a new production recovery rule unless all are true:

1. It is backed by the audit or by multiple real saved videos.
2. It uses a movement signature, not a file-specific fingerprint.
3. It has a focused test.
4. It removes or replaces an older rule, or it is part of the new final router.

## Regression Gate

The reviewed rep truth set is now the minimum gate for classifier and rep-count
changes. It must stay green before promoting more coverage:

```bash
tools/run_rep_truth_gate.sh
```

The gate runs `backend/ml/benchmark/config/rep_truth_manifest.csv` with fresh
analyzer responses and strict scoring. A row fails when either the final label
or the expected rep count is wrong.

New reviewed-contact-sheet candidates should go through the expansion queue
first:

```bash
tools/run_rep_truth_expansion.sh
```

Only rows that pass the expansion run should be promoted into the strict gate.

## Next Work Slice

Create `final_decision_router.py` without changing behavior:

1. Define `RouterContext` and `RouterDecision`.
2. Move protected evidence assignment into the new module.
3. Keep the same output for existing focused tests.
4. Leave shadow routers audit-only.
5. Run the focused router suite and the short analyzer audit.
