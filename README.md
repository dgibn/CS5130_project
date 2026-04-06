# CS5130_project

## Expose the Trading API on the internet (ngrok)

The FastAPI app lives in [`src/api.py`](src/api.py) and listens on **port 8000** by default (`0.0.0.0`).

### One command: API + ngrok together

From the **project root**:

1. **Install [ngrok](https://ngrok.com/download)** and add your authtoken (one-time):

   ```bash
   ngrok config add-authtoken YOUR_TOKEN
   ```

2. **Run** [`scripts/start-api-and-ngrok.sh`](scripts/start-api-and-ngrok.sh):

   ```bash
   ./scripts/start-api-and-ngrok.sh
   ```

   By default the script starts the API with **`conda run -n cs5130 python api.py`** (no need to `conda activate` first). It looks for `conda` on your `PATH`, and if missing tries to `source` `conda.sh` from common install locations (`~/miniconda3`, `~/miniforge3`, Homebrew Minicask paths, etc.).

   Then it waits until `/api/tickers` responds and runs **`ngrok http 8000`** in the foreground. **Ctrl+C** stops ngrok and terminates the API process.

   **Conda / Python options:**

   | Variable | Default | Meaning |
   |----------|---------|---------|
   | `CONDA_ENV` | `cs5130` | Env name passed to `conda run -n` |
   | `USE_CONDA` | `1` | Set to `0` to skip conda and use `PYTHON` instead |
   | `PYTHON` | `python` | Used only when `USE_CONDA=0` |
   | `PORT` | `8000` | Local API port |
   | `DQN_CHECKPOINT` | (unset) | Passed through to the API process |

   Examples:

   ```bash
   # Different conda env
   CONDA_ENV=myenv ./scripts/start-api-and-ngrok.sh

   # No conda: use a venv or system Python you already activated
   USE_CONDA=0 PYTHON=python3 ./scripts/start-api-and-ngrok.sh

   # Explicit interpreter (still no conda)
   USE_CONDA=0 PYTHON="$HOME/miniconda3/envs/cs5130/bin/python" DQN_CHECKPOINT=/path/to/Q_net.pt ./scripts/start-api-and-ngrok.sh
   ```

   **Why `conda activate` failed in the script:** `activate` is a shell hook. In a non-interactive script you must either `source …/conda.sh` and then `conda activate`, or use **`conda run -n envname command`**, which the script uses so the API always runs inside the right env.

3. Copy the **HTTPS** URL from ngrok’s output and set it as the `API` base URL in [`frontend/src/App.js`](frontend/src/App.js) (or your env). The free tier may require the `ngrok-skip-browser-warning` header; the bundled frontend already sends it.

The API loads `Q_net.pt` from **`src/` or the project root** unless `DQN_CHECKPOINT` is set.

### Manual two-terminal setup

1. **Terminal A — API** (`cd src && python api.py`).

2. **Terminal B — tunnel**:

   ```bash
   ./scripts/ngrok-tunnel.sh
   ```

   Or: `ngrok http 8000`

3. Use the public **HTTPS** URL for routes, e.g. `https://xxxx.ngrok-free.app/api/predict/AAPL`.

**Security:** The tunnel exposes your machine to the web. Use ngrok’s IP restrictions or auth if needed; do not rely on open CORS alone. For production, prefer a VPS or PaaS with HTTPS and rate limiting.

**Alternatives:** [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/) (`cloudflared tunnel`), or deploy behind nginx on a cloud VM.