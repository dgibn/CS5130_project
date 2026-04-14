# CS5130 Trading Project

Reinforcement-learning trading agents (tabular Q-learning, RNN-DQN, and Dueling DQN variants) with a FastAPI backend and a React dashboard.

---

## Environment setup

### Conda environment (Python backend)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) if you do not already have Conda.

2. Create an environment with a recent Python (3.10–3.12 work well with PyTorch):

   ```bash
   conda create -n cs5130 python=3.11 -y
   conda activate cs5130
   ```

3. From the **project root** (`CS5130_project/`), install Python dependencies:

   ```bash
   pip install -r requirements.txt
   pip install tqdm
   ```

   (`tqdm` is used by the training scripts for progress bars and is not listed in `requirements.txt`.)

4. Run training, tests, and the API from the **`backend`** folder with this environment active (`conda activate cs5130`). The repo’s helper scripts may refer to the env name **`cs5130`**; rename the env or adjust those scripts if you pick another name.

### Frontend (optional)

To run the React app locally:

```bash
cd frontend
npm install
npm start
```

---

## 1. Training each model

All training scripts read shared settings from [`backend/config.py`](backend/config.py). Run them from the **`backend`** directory so imports resolve:

```bash
cd backend
```

| Model | How to run | Output artifact |
|--------|------------|-----------------|
| **Tabular Q-learning** | Set `MODE = "qlearning"` in [`config.py`](backend/config.py), then `python train.py` | `Q_table.pickle` |
| **RNN-DQN** | Set `MODE = "rnn"` in `config.py`, then `python train.py` | `rnn_q_network.pt` (best checkpoint) |
| **3-action Dueling DQN** | Set `MODE = "qnet"` in `config.py`, then `python train.py` | `train_qNET()` saves weights (see note below) |
| **11-action DQN + volume head** | `python train_volume.py` (no `MODE`; dedicated script) | `Q_net_final.pt`, and periodic `Q_net_best{episode}.pt` during training |

**Notes:**

- `train.py` selects the routine via **`MODE`**: `qlearning`, `rnn`, or `qnet` only ([`train.py`](backend/train.py) `if __name__ == "__main__"`).
- Volume training is **`train_volume.py`** only; do not add `qnet_volume` to `train.py`.
- `train.py` uses a module-local `PERIOD = "5y"` for its download window (it overrides `config.PERIOD` for that script). The 3-action DQN path saves to a path defined inside [`train.py`](backend/train.py) (`train_qNET`); adjust that path or copy the resulting `.pt` next to `test.py` and point **`QNET_CHECKPOINT`** at it when testing (see below).

---

## 2. Testing each model

Tests are driven by **`MODE`** in [`backend/config.py`](backend/config.py), including an extra mode for the volume-trained network:

| `MODE` | What it evaluates | Default checkpoint (override with env var) |
|--------|-------------------|-------------------------------------------|
| `qlearning` | Tabular policy vs test split | Requires `Q_table.pickle` in the working directory |
| `rnn` | RNN-DQN | `RNN_CHECKPOINT` (default `rnn_q_network.pt`) |
| `qnet` | 3-action `Q_network.DuelingDQN` | `QNET_CHECKPOINT` (default `Q_net4.pt`) |
| `qnet_volume` | 11-action `DQN.DuelingDQN` from `train_volume.py` | `DQN_CHECKPOINT` (default `Q_net_best.pt`) |

Run from **`backend`**:

```bash
cd backend
python test.py
```

`test.py` writes aggregated rows to **`cash_history.csv`** at the project root (or current working directory), with a **`Method`** column set to the current `MODE`.

**Environment variables (optional):** `QNET_CHECKPOINT`, `DQN_CHECKPOINT`, `RNN_CHECKPOINT` — same idea as the API’s `DQN_CHECKPOINT` for picking weights.

---

## 3. Changing basic configuration

Edit [`backend/config.py`](backend/config.py). Common knobs:

| Setting | Purpose |
|---------|---------|
| **`MODE`** | Which trainer `train.py` runs (`qlearning` / `rnn` / `qnet`) and which evaluator `test.py` runs (adds `qnet_volume`). |
| **`TICKERS`** | Universe of symbols (must stay consistent across training, testing, and API). |
| **`PERIOD`** | Yahoo Finance history length for many pipelines (note: `train.py` may override with its own `PERIOD` for downloads). |
| **`EVAL_HISTORY_PERIOD`** | Window used by the API, `test.py`, and backtest-style eval. |
| **`CASH`** | Initial cash per portfolio / episode where applicable. |
| **`TRAIN_SIZE`** | Train/test split fraction (e.g. `0.8`). |
| **`SEQ_LEN`**, **`BATCH_SIZE`**, **`NUM_EPISODES_QL`**, etc. | Model and training hyperparameters (see comments in `config.py`). |

After changing training settings, re-run the matching training script, then testing. For the **deployed web app**, the backend must serve a checkpoint compatible with [`backend/api.py`](backend/api.py) (see `DQN_CHECKPOINT` in deployment).

---

## 4. Viewing results on the frontend

The hosted dashboard is:

**[https://cs-5130-project.vercel.app/](https://cs-5130-project.vercel.app/)**

The React app calls the FastAPI service whose base URL is set in [`frontend/src/App.js`](frontend/src/App.js) as **`API`** (currently a Google Cloud Run URL). When the API is up and CORS allows the Vercel origin, the UI shows:

- Ticker selection, price history, and model **predictions**
- **Portfolio** / backtest-style summaries driven by the live API

If the page shows **API Offline** or failed requests, confirm the Cloud Run (or your) backend is deployed and healthy, and that `API` in `App.js` matches that service. Rebuild or redeploy the frontend after changing `API` if you host a static build.

---

## Local API and tunnels (optional)

For development, you can run the API locally and expose it with ngrok or similar. Older scripts may live under [`scripts/`](scripts/). The FastAPI app for this repo is [`backend/api.py`](backend/api.py); run it from `backend` (or set `PYTHONPATH` appropriately) and align the frontend `API` URL with your tunnel.
