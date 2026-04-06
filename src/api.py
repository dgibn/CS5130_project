"""
FastAPI backend for the Dueling DQN Trading API.
Loads [src/DQN.py](DQN.py) and serves predictions with action + volume head.

Checkpoint: DQN_CHECKPOINT env, else src/Q_net.pt, else project-root Q_net.pt.
If you trained with train_volume.py's 5-class volume head, weights must match DQN.py
(4 classes: 5/10/15/20 shares) or load will fail — retrain or align architectures.

Usage:
    pip install fastapi uvicorn
    cd src && python api.py
"""

from __future__ import annotations

import os
import warnings
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import ACTIONS, CASH, INPUT_DIM, SEQ_LEN, TICKERS, TRAIN_SIZE
from dataloader import FEATURE_COLS, train_test_split
from dataloader_rnn import preprocess_for_rnn
from DQN import DuelingDQN

warnings.filterwarnings("ignore")

yf.set_tz_cache_location("/tmp")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))


def resolve_checkpoint() -> str:
    """Prefer DQN_CHECKPOINT; else src/Q_net.pt; else ../Q_net.pt (project root)."""
    env = os.environ.get("DQN_CHECKPOINT")
    if env and os.path.isfile(env):
        return env
    for path in (
        os.path.join(_SRC_DIR, "Q_net.pt"),
        os.path.join(_PROJECT_ROOT, "Q_net.pt"),
    ):
        if os.path.isfile(path):
            return path
    return os.path.join(_SRC_DIR, "Q_net.pt")


CHECKPOINT = resolve_checkpoint()

STATE_SIZE = SEQ_LEN * len(FEATURE_COLS) + 2
assert len(FEATURE_COLS) == INPUT_DIM
assert STATE_SIZE == SEQ_LEN * INPUT_DIM + 2

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

app = FastAPI(title="Dueling DQN Trading API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: DuelingDQN | None = None


def load_model() -> None:
    global model
    model = DuelingDQN(STATE_SIZE, len(ACTIONS)).to(device)
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found. Tried: DQN_CHECKPOINT, "
            f"{os.path.join(_SRC_DIR, 'Q_net.pt')}, {os.path.join(_PROJECT_ROOT, 'Q_net.pt')}\n"
            f"Set DQN_CHECKPOINT=/path/to/model.pt or copy Q_net.pt into src/ or project root.\n"
            f"Train with train_volume.py (train_qNET) using DQN.DuelingDQN."
        )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"  DuelingDQN loaded from {CHECKPOINT} (device: {device})")


@app.on_event("startup")
async def startup() -> None:
    print("Starting Dueling DQN Trading API...")
    load_model()
    print("API ready!")


def get_state(df: pd.DataFrame, idx: int, cash: float, volume: int) -> torch.Tensor:
    row = df.iloc[idx - SEQ_LEN + 1 : idx + 1].values.flatten().astype(np.float32)
    portfolio = np.array([cash / CASH, float(volume) / 10.0], dtype=np.float32)
    return torch.tensor(np.concatenate([row, portfolio]), dtype=torch.float32)


def fetch_test_df(ticker: str, period: str) -> Tuple[pd.DataFrame, np.ndarray]:
    hist = yf.Ticker(ticker).history(period=period)
    hist.reset_index(inplace=True)
    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)
    prices = test_df["Close"].values
    return test_df, prices


@torch.no_grad()
def forward_policy(state_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    assert model is not None
    model.reset_noise()
    s = state_1d.unsqueeze(0).to(device)
    q_values, volume_logits = model(s)
    action_idx = int(q_values.argmax(dim=-1).item())
    vol_idx = int(volume_logits.argmax(dim=-1).item())
    pred_vol = int(model.volume_classes[vol_idx])
    return q_values.squeeze(0), volume_logits.squeeze(0), action_idx, pred_vol


def run_trading_episode(
    test_df: pd.DataFrame,
    prices: np.ndarray,
    initial_cash: float,
) -> Tuple[list, float]:
    """Step through test_df from SEQ_LEN; greedy action + volume from model."""
    cash = float(initial_cash)
    volume = 0
    action_names = ["HOLD", "BUY", "SELL"]
    portfolio: list = []

    T = len(test_df)
    if T <= SEQ_LEN + 1:
        return portfolio, initial_cash

    portfolio.append({
        "day": 0,
        "value": round(initial_cash, 2),
        "action": "START",
        "predicted_volume": 0,
        "price": round(float(prices[SEQ_LEN]), 2),
        "cash": round(cash, 2),
        "shares": volume,
    })

    for t in range(SEQ_LEN, T - 1):
        state = get_state(test_df[FEATURE_COLS], t, cash, volume)
        _, _, action_idx, pred_shares = forward_policy(state)
        price_now = float(prices[t])

        if action_idx == 1 and cash >= price_now:
            sh = min(pred_shares, int(cash // price_now))
            cash -= sh * price_now
            volume += sh
        elif action_idx == 2 and volume > 0:
            sh = min(pred_shares, volume)
            cash += sh * price_now
            volume -= sh

        pv = cash + volume * price_now
        portfolio.append({
            "day": len(portfolio),
            "value": round(pv, 2),
            "action": action_names[action_idx],
            "predicted_volume": pred_shares,
            "price": round(price_now, 2),
            "cash": round(cash, 2),
            "shares": volume,
        })

    return portfolio, portfolio[-1]["value"]


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────


@app.get("/api/tickers")
async def get_tickers():
    return {"tickers": list(TICKERS)}


@app.get("/api/price/{ticker}")
async def get_price(ticker: str, period: str = "6mo"):
    ticker = ticker.upper()
    try:
        hist = yf.Ticker(ticker).history(period=period)
        hist.reset_index(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    data = []
    for _, row in hist.iterrows():
        data.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        })

    current = data[-1]["close"]
    prev = data[-2]["close"] if len(data) > 1 else current
    change_pct = round((current - prev) / prev * 100, 2)

    return {
        "ticker": ticker,
        "current_price": current,
        "change_pct": change_pct,
        "history": data,
    }


@app.get("/api/predict/{ticker}")
async def get_prediction(
    ticker: str,
    cash: float | None = None,
    shares: int | None = None,
):
    """
    Next-step signal: HOLD/BUY/SELL and predicted trade volume (share bucket).
    Optional query params: cash (default CASH), shares = shares held (default 0).
    """
    ticker = ticker.upper()
    c = float(CASH if cash is None else cash)
    vol = 0 if shares is None else int(shares)

    try:
        test_df, prices = fetch_test_df(ticker, "1y")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

    if len(test_df) <= SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough rows after preprocessing (need > {SEQ_LEN})",
        )

    idx = len(test_df) - 1
    state = get_state(test_df[FEATURE_COLS], idx, c, vol)
    q_t, vol_logits_t, action_idx, pred_volume = forward_policy(state)

    q_np = q_t.cpu().numpy()
    exp_q = np.exp(q_np - q_np.max())
    softmax_q = exp_q / exp_q.sum()
    confidence = float(softmax_q[action_idx] * 100)

    vol_probs = F.softmax(vol_logits_t, dim=-1).cpu().numpy()
    action_names = ["HOLD", "BUY", "SELL"]

    return {
        "ticker": ticker,
        "action": action_names[action_idx],
        "action_index": action_idx,
        "predicted_volume": pred_volume,
        "confidence": round(confidence, 1),
        "q_values": {
            "hold": round(float(q_np[0]), 4),
            "buy": round(float(q_np[1]), 4),
            "sell": round(float(q_np[2]), 4),
        },
        "volume_logits": vol_logits_t.cpu().numpy().tolist(),
        "volume_probs": {str(model.volume_classes[i]): round(float(vol_probs[i]), 4) for i in range(len(model.volume_classes))},
        "price": round(float(prices[idx]), 2),
        "cash_used_in_state": c,
        "shares_held_in_state": vol,
    }


@app.get("/api/backtest/{ticker}")
async def get_backtest(ticker: str, initial_cash: float = 10000):
    ticker = ticker.upper()
    try:
        hist = yf.Ticker(ticker).history(period="2y")
        hist.reset_index(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    split = int(len(hist) * TRAIN_SIZE)
    test_raw = hist.iloc[split:].reset_index(drop=True)
    test_df, _ = preprocess_for_rnn(test_raw)
    prices = test_df["Close"].values

    portfolio, dqn_final = run_trading_episode(test_df, prices, initial_cash)

    if not portfolio:
        raise HTTPException(status_code=400, detail="Insufficient data for backtest")

    first_price = float(prices[SEQ_LEN])
    last_price = float(prices[-1])
    bh_shares = int(initial_cash // first_price)
    bh_residual = initial_cash - bh_shares * first_price
    bh_final = bh_shares * last_price + bh_residual

    return {
        "ticker": ticker,
        "initial_cash": initial_cash,
        "dqn_final": round(float(dqn_final), 2),
        "bh_final": round(float(bh_final), 2),
        "dqn_return": round((float(dqn_final) / initial_cash - 1) * 100, 2),
        "bh_return": round((bh_final / initial_cash - 1) * 100, 2),
        "test_days": len(portfolio),
        "portfolio": portfolio,
    }


@app.get("/api/simulate/{ticker}")
async def simulate_forward(
    ticker: str,
    days: int = 10,
    initial_cash: float = 10000,
    n_simulations: int = 50,
):
    ticker = ticker.upper()
    if days < 1 or days > 60:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 60")

    try:
        hist = yf.Ticker(ticker).history(period="1y")
        hist.reset_index(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    close = hist["Close"].values
    log_returns = np.diff(np.log(close))
    mu = float(np.mean(log_returns))
    sigma = float(np.std(log_returns))
    last_price = float(close[-1])

    all_paths = []
    for _ in range(n_simulations):
        prices_path = [last_price]
        for _d in range(days):
            shock = np.random.normal(mu, sigma)
            prices_path.append(prices_path[-1] * np.exp(shock))
        all_paths.append(prices_path)

    hist_tail = hist.tail(60).copy()
    all_portfolios: list = []
    all_actions: list = []

    for path in all_paths:
        cash = float(initial_cash)
        shares_held = 0
        pv_list = [initial_cash]
        actions_list: list = []

        future_dates = pd.bdate_range(
            start=hist["Date"].iloc[-1] + pd.Timedelta(days=1),
            periods=days,
        )
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Open": path[1:],
            "High": [p * 1.01 for p in path[1:]],
            "Low": [p * 0.99 for p in path[1:]],
            "Close": path[1:],
            "Volume": [int(hist["Volume"].mean())] * days,
        })

        combined_df = pd.concat([
            hist_tail[["Date", "Open", "High", "Low", "Close", "Volume"]],
            future_df,
        ], ignore_index=True)

        try:
            full_df, _ = preprocess_for_rnn(combined_df)
            n_steps = len(full_df)
            if n_steps <= SEQ_LEN + 2:
                raise ValueError("too short after preprocess")

            future_start = max(SEQ_LEN, n_steps - days - 1)
            prices_a = full_df["Close"].values

            for d in range(days):
                t = future_start + d
                if t >= n_steps - 1:
                    break
                state = get_state(full_df[FEATURE_COLS], t, cash, shares_held)
                _, _, action_idx, _pred = forward_policy(state)
                price_now = float(prices_a[t])

                if action_idx == 1 and cash >= price_now:
                    sh = min(_pred, int(cash // price_now))
                    cash -= sh * price_now
                    shares_held += sh
                elif action_idx == 2 and shares_held > 0:
                    sh = min(_pred, shares_held)
                    cash += sh * price_now
                    shares_held -= sh

                pv = cash + shares_held * price_now
                pv_list.append(pv)
                actions_list.append(["HOLD", "BUY", "SELL"][action_idx])

            while len(pv_list) <= days:
                pv_list.append(pv_list[-1])
            while len(actions_list) < days:
                actions_list.append("HOLD")

        except Exception:
            pv_list = [initial_cash] * (days + 1)
            actions_list = ["HOLD"] * days

        all_portfolios.append(pv_list[: days + 1])
        all_actions.append(actions_list[:days])

    portfolios_arr = np.array(all_portfolios)
    avg_pv = portfolios_arr.mean(axis=0).tolist()
    best_pv = portfolios_arr.max(axis=0).tolist()
    worst_pv = portfolios_arr.min(axis=0).tolist()
    p25_pv = np.percentile(portfolios_arr, 25, axis=0).tolist()
    p75_pv = np.percentile(portfolios_arr, 75, axis=0).tolist()

    common_actions = []
    for d in range(days):
        day_actions = [all_actions[s][d] for s in range(n_simulations) if d < len(all_actions[s])]
        common_actions.append(
            Counter(day_actions).most_common(1)[0][0] if day_actions else "HOLD"
        )

    paths_arr = np.array(all_paths)
    avg_price = paths_arr.mean(axis=0).tolist()
    best_price = paths_arr.max(axis=0).tolist()
    worst_price = paths_arr.min(axis=0).tolist()

    result_days = []
    for d in range(days + 1):
        result_days.append({
            "day": d,
            "avg_portfolio": round(avg_pv[d], 2),
            "best_portfolio": round(best_pv[d], 2),
            "worst_portfolio": round(worst_pv[d], 2),
            "p25_portfolio": round(p25_pv[d], 2),
            "p75_portfolio": round(p75_pv[d], 2),
            "avg_price": round(avg_price[d], 2),
            "best_price": round(best_price[d], 2),
            "worst_price": round(worst_price[d], 2),
            "action": common_actions[d - 1] if d > 0 else "START",
        })

    final_avg = avg_pv[-1]
    return {
        "ticker": ticker,
        "initial_cash": initial_cash,
        "days": days,
        "simulations": n_simulations,
        "current_price": last_price,
        "final_avg_portfolio": round(final_avg, 2),
        "final_best_portfolio": round(best_pv[-1], 2),
        "final_worst_portfolio": round(worst_pv[-1], 2),
        "expected_return": round((final_avg / initial_cash - 1) * 100, 2),
        "trajectory": result_days,
    }


@app.get("/api/portfolio/summary")
async def get_portfolio_summary():
    results = {}
    total_dqn = 0.0
    total_bh = 0.0

    for ticker in TICKERS:
        try:
            hist = yf.Ticker(ticker).history(period="2y")
            hist.reset_index(inplace=True)
            if hist.empty:
                raise ValueError("empty hist")
            split = int(len(hist) * TRAIN_SIZE)
            test_raw = hist.iloc[split:].reset_index(drop=True)
            test_df, _ = preprocess_for_rnn(test_raw)
            prices = test_df["Close"].values

            _, dqn_final = run_trading_episode(test_df, prices, 2000.0)

            first_price = float(prices[SEQ_LEN])
            last_price = float(prices[-1])
            bh_shares = int(2000 // first_price)
            bh_residual = 2000 - bh_shares * first_price
            bh_final = bh_shares * last_price + bh_residual

            results[ticker] = {
                "dqn": round(float(dqn_final), 2),
                "bh": round(float(bh_final), 2),
                "dqn_ret": round((float(dqn_final) / 2000 - 1) * 100, 2),
                "bh_ret": round((bh_final / 2000 - 1) * 100, 2),
            }
            total_dqn += float(dqn_final)
            total_bh += bh_final

        except Exception as e:
            results[ticker] = {
                "dqn": 2000,
                "bh": 2000,
                "dqn_ret": 0,
                "bh_ret": 0,
                "error": str(e),
            }

    return {
        "tickers": results,
        "total_dqn": round(total_dqn, 2),
        "total_bh": round(total_bh, 2),
        "total_dqn_return": round((total_dqn / 10000 - 1) * 100, 2),
        "total_bh_return": round((total_bh / 10000 - 1) * 100, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
