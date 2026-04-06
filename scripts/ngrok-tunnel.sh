#!/usr/bin/env bash
# Tunnel the local Trading API to the internet via ngrok.
#
# To start API + ngrok together, use instead:
#   ./scripts/start-api-and-ngrok.sh
#
# This script is for when the API is already running.
#
# 1. Start the API (separate terminal):
#      cd src && python api.py
#    Or: cd src && uvicorn api:app --host 0.0.0.0 --port 8000
#
# 2. Run this script (install ngrok first: https://ngrok.com/download):
#      ./scripts/ngrok-tunnel.sh
#
# Optional: PORT=8080 ./scripts/ngrok-tunnel.sh
# Authenticate once: ngrok config add-authtoken <token>  (from https://dashboard.ngrok.com)

set -euo pipefail
PORT="${PORT:-8000}"

if ! command -v ngrok &>/dev/null; then
  echo "ngrok not found. Install: https://ngrok.com/download"
  echo "Then: ngrok config add-authtoken <your_token>"
  exit 1
fi

echo "Forwarding public HTTPS -> http://127.0.0.1:${PORT}"
echo "Ensure the API is already listening on port ${PORT}."
exec ngrok http "$PORT"
