# CS5130_project

## Expose the Trading API on the internet (ngrok)

The FastAPI app lives in [`src/api.py`](src/api.py) and listens on **port 8000** by default (`0.0.0.0`).

1. **Install [ngrok](https://ngrok.com/download)** and add your authtoken (one-time):

   ```bash
   ngrok config add-authtoken YOUR_TOKEN
   ```

2. **Start the API** (`cd src && python api.py`). It loads `Q_net.pt` from **`src/` or the project root** (or set `DQN_CHECKPOINT` to an explicit path).

   ```bash
   cd src
   python api.py
   ```

3. **In another terminal**, start the tunnel:

   ```bash
   ./scripts/ngrok-tunnel.sh
   ```

   Or manually: `ngrok http 8000`

4. ngrok prints a public **HTTPS** URL (e.g. `https://xxxx.ngrok-free.app`). Call your routes there, e.g. `https://xxxx.ngrok-free.app/api/predict/AAPL`.

**Security:** The tunnel exposes your machine to the web. Use ngrok’s IP restrictions or auth if needed; do not rely on open CORS alone. For production, prefer a VPS or PaaS with HTTPS and rate limiting.

**Alternatives:** [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/) (`cloudflared tunnel`), or deploy behind nginx on a cloud VM.