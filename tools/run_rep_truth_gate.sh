#!/usr/bin/env bash
set -euo pipefail

cd /Users/josephkamil/Downloads/formcheck_main_merge
source /Users/josephkamil/Downloads/formcheck_real_inference_updated/backend/.venv/bin/activate

PYTHONPATH=backend python backend/ml/benchmark/run_analyzer_audit.py \
  --candidates backend/ml/benchmark/config/rep_truth_manifest.csv \
  --out backend/ml/benchmark/results/rep_truth_gate_latest.csv \
  --save-responses backend/ml/benchmark/results/rep_truth_gate_responses \
  --timeout 180 \
  --fresh \
  --strict
