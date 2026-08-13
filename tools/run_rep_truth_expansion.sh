#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/Users/josephkamil/Downloads/formcheck_main_merge"
VENV_ACTIVATE="/Users/josephkamil/Downloads/formcheck_real_inference_updated/backend/.venv/bin/activate"
HOST="${FORMCHECK_BACKEND_HOST:-127.0.0.1}"
PORT="${FORMCHECK_BACKEND_PORT:-8000}"
LOG_FILE="$REPO_DIR/backend/ml/benchmark/results/rep_truth_backend.log"

cd "$REPO_DIR"
source "$VENV_ACTIVATE"
mkdir -p "$(dirname "$LOG_FILE")"

run_audit() {
  PYTHONPATH="$REPO_DIR/backend" python "$REPO_DIR/backend/ml/benchmark/run_analyzer_audit.py" \
    --candidates "$REPO_DIR/backend/ml/benchmark/config/rep_truth_expansion_manifest.csv" \
    --out "$REPO_DIR/backend/ml/benchmark/results/rep_truth_expansion_latest.csv" \
    --save-responses "$REPO_DIR/backend/ml/benchmark/results/rep_truth_expansion_responses" \
    --timeout 180 \
    --fresh \
    --strict
}

if [[ "${FORMCHECK_USE_EXISTING_BACKEND:-0}" == "1" ]]; then
  echo "Using existing backend on $HOST:$PORT"
  run_audit
  exit 0
fi

echo "Checking repo code loaded by the backend..."
PYTHONPATH="$REPO_DIR/backend" python - <<'PY'
import app.ml.rep_detector as rep_detector

print(rep_detector.__file__)
if not hasattr(rep_detector, "recover_long_overhead_squat_clip"):
    raise SystemExit("Expected rep detector recovery code is not loaded")
PY

if command -v lsof >/dev/null 2>&1; then
  existing_pids="$(lsof -ti "tcp:$PORT" || true)"
  if [[ -n "$existing_pids" ]]; then
    echo "Stopping existing backend on port $PORT: $existing_pids"
    kill $existing_pids || true
    sleep 1
  fi
fi

cd "$REPO_DIR/backend"
echo "Starting fresh backend on $HOST:$PORT"
MPLCONFIGDIR=/private/tmp/formcheck-mpl-cache \
PYTHONPATH="$REPO_DIR/backend" \
uvicorn app.main:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
backend_pid=$!

cleanup() {
  kill "$backend_pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in {1..120}; do
  if ! kill -0 "$backend_pid" >/dev/null 2>&1; then
    echo "Backend exited before it was ready. Recent log:"
    tail -80 "$LOG_FILE" || true
    exit 1
  fi

  if python -c "import urllib.request; urllib.request.urlopen('http://$HOST:$PORT/docs', timeout=1).read(1)" >/dev/null 2>&1; then
    ready=1
    break
  fi

  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "Backend did not become ready. Recent log:"
  tail -80 "$LOG_FILE" || true
  exit 1
fi

echo "Backend is ready. Running expansion audit..."
cd "$REPO_DIR"
run_audit
