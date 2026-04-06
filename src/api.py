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
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler

from config import ACTIONS, CASH, INPUT_DIM, SEQ_LEN, TICKERS
from dataloader import FEATURE_COLS, _add_features
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


# Longer periods are not always more reliable with yfinance; try several until SEQ_LEN rows remain.
_INFERENCE_PERIODS = ("max", "10y", "5y", "2y", "1y", "6mo")


def _normalize_yfinance_hist(hist: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns so Close/Volume are accessible (single-ticker edge cases)."""
    if hist.empty or not isinstance(hist.columns, pd.MultiIndex):
        return hist
    for level in (0, -1):
        names = hist.columns.get_level_values(level)
        if "Close" in names and "Volume" in names:
            out = hist.copy()
            out.columns = names
            return out
    return hist


def _preprocess_for_inference(hist: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Like preprocess_for_rnn but replaces inf before scaling (API inference only)."""
    df = _add_features(hist.copy())
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)
    scaler = StandardScaler()
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    return df, scaler


def fetch_inference_df(ticker: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Full history for live /predict (no train–test split).

    Tries multiple yfinance periods until preprocessing yields len >= SEQ_LEN.
    """
    import logging
    logger = logging.getLogger("uvicorn.error")

    last_exc: Exception | None = None
    saw_nonempty = False

    for period in _INFERENCE_PERIODS:
        try:
            hist = yf.Ticker(ticker).history(period=period)
            hist = hist.reset_index(drop=False)
            hist = _normalize_yfinance_hist(hist)
            logger.info(f"[predict] {ticker} period={period} rows={len(hist)} cols={list(hist.columns)[:6]}")
            if hist.empty:
                continue
            saw_nonempty = True
            if "Close" not in hist.columns or "Volume" not in hist.columns:
                logger.warning(f"[predict] {ticker} period={period}: missing Close/Volume in {list(hist.columns)}")
                continue
            test_df, _ = _preprocess_for_inference(hist)
            logger.info(f"[predict] {ticker} period={period} after preprocess rows={len(test_df)}")
            if len(test_df) >= SEQ_LEN:
                return test_df, test_df["Close"].values
        except Exception as e:
            logger.warning(f"[predict] {ticker} period={period} exception: {e}")
            last_exc = e
            continue

    if not saw_nonempty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")
    if last_exc is not None:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {last_exc}")
    raise HTTPException(
        status_code=400,
        detail=(
            f"Not enough rows after preprocessing (need at least {SEQ_LEN}). "
            "Tried multiple history periods."
        ),
    )


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


def _execute_backtest(ticker: str, initial_cash: float, test_days: int) -> dict:
    """
    Shared pipeline for /api/backtest and /api/portfolio/summary.
    Raises ValueError on failure (message used for HTTP detail or per-ticker error).
    """
    t = ticker.upper()
    try:
        hist = yf.Ticker(t).history(period="5y")
        hist.reset_index(inplace=True)
    except Exception as e:
        raise ValueError(f"Failed to fetch {t}: {e}") from e

    if hist.empty:
        raise ValueError(f"No data for {t}")

    needed_raw = test_days + SEQ_LEN + 60
    test_raw = hist.tail(needed_raw).reset_index(drop=True)
    test_df, _ = preprocess_for_rnn(test_raw)
    prices = test_df["Close"].values

    portfolio, dqn_final = run_trading_episode(test_df, prices, initial_cash)
    if not portfolio:
        raise ValueError("Insufficient data for backtest")

    first_price = float(prices[SEQ_LEN])
    last_price = float(prices[-1])
    bh_shares = int(initial_cash // first_price)
    bh_residual = initial_cash - bh_shares * first_price
    bh_final = bh_shares * last_price + bh_residual

    return {
        "ticker": t,
        "initial_cash": initial_cash,
        "dqn_final": round(float(dqn_final), 2),
        "bh_final": round(float(bh_final), 2),
        "dqn_return": round((float(dqn_final) / initial_cash - 1) * 100, 2),
        "bh_return": round((bh_final / initial_cash - 1) * 100, 2),
        "test_days": len(portfolio),
        "portfolio": portfolio,
    }


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
        test_df, prices = fetch_inference_df(ticker)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

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
async def get_backtest(ticker: str, initial_cash: float = 10000, test_days: int = 100):
    try:
        return _execute_backtest(ticker, initial_cash, test_days)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("No data for"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@app.get("/api/portfolio/summary")
async def get_portfolio_summary(initial_cash: float = 2000, test_days: int = 100):
    results: dict = {}
    total_dqn = 0.0
    total_bh = 0.0
    capital_base = float(len(TICKERS) * initial_cash)

    for ticker in TICKERS:
        try:
            r = _execute_backtest(ticker, initial_cash, test_days)
            results[ticker] = {
                "dqn": r["dqn_final"],
                "bh": r["bh_final"],
                "dqn_ret": r["dqn_return"],
                "bh_ret": r["bh_return"],
            }
            total_dqn += float(r["dqn_final"])
            total_bh += float(r["bh_final"])
        except ValueError as e:
            results[ticker] = {
                "dqn": round(initial_cash, 2),
                "bh": round(initial_cash, 2),
                "dqn_ret": 0.0,
                "bh_ret": 0.0,
                "error": str(e),
            }

    return {
        "tickers": results,
        "initial_cash_per_ticker": initial_cash,
        "test_days_param": test_days,
        "total_dqn": round(total_dqn, 2),
        "total_bh": round(total_bh, 2),
        "total_dqn_return": round((total_dqn / capital_base - 1) * 100, 2) if capital_base else 0.0,
        "total_bh_return": round((total_bh / capital_base - 1) * 100, 2) if capital_base else 0.0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
