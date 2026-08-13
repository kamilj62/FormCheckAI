#!/usr/bin/env bash
set -euo pipefail

cd /Users/josephkamil/Downloads/formcheck_main_merge
source /Users/josephkamil/Downloads/formcheck_real_inference_updated/backend/.venv/bin/activate

echo "Using rep detector from:"
PYTHONPATH=/Users/josephkamil/Downloads/formcheck_main_merge/backend python - <<'PY'
import app.ml.rep_detector as rep_detector
print(rep_detector.__file__)
print("overhead recovery:", hasattr(rep_detector, "recover_long_overhead_squat_clip"))
PY

if command -v lsof >/dev/null 2>&1; then
  existing_pids="$(lsof -ti tcp:8000 || true)"
  if [ -n "$existing_pids" ]; then
    echo "Stopping existing backend on port 8000: $existing_pids"
    kill $existing_pids || true
    sleep 1
  fi
fi

cd /Users/josephkamil/Downloads/formcheck_main_merge/backend
export MPLCONFIGDIR=/private/tmp/formcheck-mpl-cache
export PYTHONPATH=/Users/josephkamil/Downloads/formcheck_main_merge/backend

exec uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
