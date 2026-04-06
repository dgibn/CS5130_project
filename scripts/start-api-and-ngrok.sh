#!/usr/bin/env bash
# Start the Trading API (src/api.py) and ngrok in one terminal.
# From the project root:
#   ./scripts/start-api-and-ngrok.sh
#
# Conda: by default uses env "cs5130" via `conda run` (no `conda activate` needed).
#   USE_CONDA=0 ./scripts/start-api-and-ngrok.sh     # use PYTHON only (system / venv)
#   CONDA_ENV=myenv ./scripts/start-api-and-ngrok.sh # different env name
#
# If `conda` is not on PATH, common install locations are tried (miniconda3, anaconda3, miniforge).
#
# Other env:
#   PORT, PYTHON (when USE_CONDA=0), DQN_CHECKPOINT

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
PYTHON="${PYTHON:-python}"
export DQN_CHECKPOINT="${DQN_CHECKPOINT:-}"
USE_CONDA="${USE_CONDA:-1}"
CONDA_ENV="${CONDA_ENV:-cs5130}"

cd "$ROOT"

# conda activate does not work in scripts unless conda is hooked into this shell.
_load_conda_sh() {
  if command -v conda &>/dev/null; then
    return 0
  fi
  local f
  for f in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh" \
    "/usr/local/Caskroom/miniconda/base/etc/profile.d/conda.sh"; do
    if [[ -f "$f" ]]; then
      # shellcheck source=/dev/null
      source "$f"
      return 0
    fi
  done
  return 1
}

_run_api() {
  cd "$ROOT/src"
  if [[ "$USE_CONDA" == "1" ]]; then
    if ! command -v conda &>/dev/null; then
      echo "conda not found on PATH. Tried sourcing conda.sh from common install dirs." >&2
      echo "Fix: add Miniconda/Anaconda to PATH, or run:" >&2
      echo "  USE_CONDA=0 PYTHON=/path/to/your/python $0" >&2
      exit 1
    fi
    exec conda run -n "$CONDA_ENV" --no-capture-output python api.py
  else
    exec "$PYTHON" api.py
  fi
}

if [[ "$USE_CONDA" == "1" ]]; then
  _load_conda_sh || true
fi

if ! command -v ngrok &>/dev/null; then
  echo "ngrok not found. Install: https://ngrok.com/download"
  echo "Then: ngrok config add-authtoken <your_token>"
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    echo ""
    echo "Stopping API (pid $API_PID)..."
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$USE_CONDA" == "1" ]]; then
  echo "Starting API: conda run -n ${CONDA_ENV} python api.py (port $PORT)"
else
  echo "Starting API: $PYTHON api.py (port $PORT)"
fi
(
  _run_api
) &
API_PID=$!

echo "Waiting for http://127.0.0.1:${PORT}/api/tickers ..."
ready=0
for _ in $(seq 1 90); do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API process exited before becoming ready. Check errors above or run: cd src && python api.py"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/api/tickers" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "Timed out waiting for API on port $PORT"
  exit 1
fi

echo "API is up. Starting ngrok (Ctrl+C stops both ngrok and the API)."
echo "Use the HTTPS URL ngrok prints for your frontend API base URL."
ngrok http "$PORT"
