"""
FastAPI backend for the Dueling DQN Trading API.
Loads [src/DQN.py](DQN.py) and serves predictions with action + volume head.

Checkpoint: DQN_CHECKPOINT env, else src/Q_net_best.pt, else project-root Q_net_best.pt.
Must match train_volume.train_qNET: 11 combined (action × volume) Q-outputs and the same
state layout (portfolio volume scaled by /20 like training). Auxiliary volume head: 4 buckets.

Usage:
    pip install fastapi uvicorn
    cd src && python api.py
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler

from config import CASH, EVAL_HISTORY_PERIOD, INPUT_DIM, SEQ_LEN, TICKERS, TRAIN_SIZE
from dataloader import FEATURE_COLS, _add_features, train_test_split
from dataloader_rnn import preprocess_for_rnn
from DQN import DuelingDQN

warnings.filterwarnings("ignore")

yf.set_tz_cache_location("/tmp")

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, ".."))


def resolve_checkpoint() -> str:
    """Prefer DQN_CHECKPOINT; else src/Q_net_best.pt; else ../Q_net_best.pt (project root)."""
    env = os.environ.get("DQN_CHECKPOINT")
    if env and os.path.isfile(env):
        return env
    for path in (
        os.path.join(_SRC_DIR, "Q_net_best.pt"),
        os.path.join(_PROJECT_ROOT, "Q_net_best.pt"),
    ):
        if os.path.isfile(path):
            return path
    return os.path.join(_SRC_DIR, "Q_net_best.pt")


CHECKPOINT = resolve_checkpoint()

STATE_SIZE = SEQ_LEN * len(FEATURE_COLS) + 2
assert len(FEATURE_COLS) == INPUT_DIM
assert STATE_SIZE == SEQ_LEN * INPUT_DIM + 2

# Same flattened action × volume space as train_volume.train_qNET (NUM_COMBINED = 11)
VOL_CLASSES = [1, 5, 10, 15, 20]
COMBINED_ACTIONS: list[tuple[int, int]] = [(0, 0)]
for v in VOL_CLASSES:
    COMBINED_ACTIONS.append((1, v))
    COMBINED_ACTIONS.append((2, v))
NUM_COMBINED = len(COMBINED_ACTIONS)

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
    model = DuelingDQN(STATE_SIZE, NUM_COMBINED).to(device)
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found. Tried: DQN_CHECKPOINT, "
            f"{os.path.join(_SRC_DIR, 'Q_net_best.pt')}, {os.path.join(_PROJECT_ROOT, 'Q_net_best.pt')}\n"
            f"Set DQN_CHECKPOINT=/path/to/model.pt or copy Q_net_best.pt into src/ or project root.\n"
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
    # Match train_volume.get_state (max volume in COMBINED_ACTIONS is 20)
    portfolio = np.array([cash / CASH, float(volume) / 20.0], dtype=np.float32)
    return torch.tensor(np.concatenate([row, portfolio]), dtype=torch.float32)


def get_legal_mask(cash: float, volume: int, price: float) -> torch.Tensor:
    """Same as train_volume: mask illegal combined actions before argmax."""
    mask = torch.ones(NUM_COMBINED, dtype=torch.bool)
    for i, (act, vol) in enumerate(COMBINED_ACTIONS):
        if act == 1 and cash < price * vol:
            mask[i] = False
        elif act == 2 and volume < vol:
            mask[i] = False
    return mask


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
def forward_policy(
    state_1d: torch.Tensor,
    cash: float,
    volume: int,
    price: float,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
    """
    Greedy policy: masked argmax on Q-values picks the combined action (hold vs buy/sell slot).
    Trade size for BUY/SELL comes from the auxiliary volume head (argmax on volume_logits
    mapped through model.volume_classes); HOLD uses 0 shares.
    """
    assert model is not None
    model.reset_noise()
    s = state_1d.unsqueeze(0).to(device)
    q_values, volume_logits = model(s)
    q_values = q_values.squeeze(0)
    vol_logits_1d = volume_logits.squeeze(0)
    mask = get_legal_mask(cash, volume, price).to(device)
    q_masked = q_values.clone()
    q_masked[~mask] = float("-inf")
    combined_idx = int(q_masked.argmax().item())
    act, _ = COMBINED_ACTIONS[combined_idx]
    vol_idx = int(vol_logits_1d.argmax().item())
    pred_shares_aux = int(model.volume_classes[vol_idx])
    pred_shares = 0 if act == 0 else pred_shares_aux
    return q_values, vol_logits_1d, combined_idx, act, pred_shares


def run_trading_episode(
    test_df: pd.DataFrame,
    prices: np.ndarray,
    initial_cash: float,
    days_to_simulate: Optional[int] = None,
) -> Tuple[list, float, int]:
    """
    Step through test_df from SEQ_LEN.

    If days_to_simulate is set, matches test.py test_Qnet_single: iterate up to that
    many steps with idx = SEQ_LEN + step, break when idx >= T - 1.

    If None, walks all timesteps from SEQ_LEN to T - 2 (legacy).

    Returns (portfolio, final_value, last_t) where last_t is the last bar index used
    for B&H valuation on the same horizon.
    """
    cash = float(initial_cash)
    volume = 0
    action_names = ["HOLD", "BUY", "SELL"]
    portfolio: list = []

    T = len(test_df)
    if T <= SEQ_LEN + 1:
        return [], float(initial_cash), SEQ_LEN

    portfolio.append({
        "day": 0,
        "value": round(initial_cash, 2),
        "action": "START",
        "predicted_volume": 0,
        "price": round(float(prices[SEQ_LEN]), 2),
        "cash": round(cash, 2),
        "shares": volume,
    })

    if days_to_simulate is None:
        t_iter = range(SEQ_LEN, T - 1)
    else:
        t_iter = []
        for step in range(days_to_simulate):
            tt = SEQ_LEN + step
            if tt >= T - 1:
                break
            t_iter.append(tt)

    last_t = SEQ_LEN
    for t in t_iter:
        state = get_state(test_df[FEATURE_COLS], t, cash, volume)
        price_now = float(prices[t])
        _, _, _, act, pred_shares = forward_policy(state, cash, volume, price_now)

        if act == 1 and cash >= price_now:
            sh = min(pred_shares, int(cash // price_now))
            cash -= sh * price_now
            volume += sh
        elif act == 2 and volume > 0:
            sh = min(pred_shares, volume)
            cash += sh * price_now
            volume -= sh

        pv = cash + volume * price_now
        last_t = t
        portfolio.append({
            "day": len(portfolio),
            "value": round(pv, 2),
            "action": action_names[act],
            "predicted_volume": pred_shares,
            "price": round(price_now, 2),
            "cash": round(cash, 2),
            "shares": volume,
        })

    final_val = portfolio[-1]["value"] if len(portfolio) > 1 else float(initial_cash)
    return portfolio, float(final_val), last_t


def _execute_backtest(ticker: str, initial_cash: float) -> dict:
    """
    Same evaluation window as test.py test_Qnet_single:
    yfinance period=EVAL_HISTORY_PERIOD, chronological train/test split (TRAIN_SIZE), preprocess
    test split only, then days_to_simulate = max(240, n_data - 1 - SEQ_LEN) steps.
    """
    t = ticker.upper()
    try:
        hist = yf.Ticker(t).history(period=EVAL_HISTORY_PERIOD)
    except Exception as e:
        raise ValueError(f"Failed to fetch {t}: {e}") from e

    if hist.empty:
        raise ValueError(f"No data for {t}")

    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)
    prices = test_df["Close"].values
    n_data = len(test_df)

    if n_data <= SEQ_LEN + 1:
        raise ValueError("Insufficient data for backtest after preprocess")

    days_to_simulate = max(240, n_data - 1 - SEQ_LEN)

    portfolio, dqn_final, last_t = run_trading_episode(
        test_df, prices, initial_cash, days_to_simulate=days_to_simulate
    )
    if not portfolio:
        raise ValueError("Insufficient data for backtest")

    first_price = float(prices[SEQ_LEN])
    last_price = float(prices[last_t])
    bh_shares = int(initial_cash // first_price)
    bh_residual = initial_cash - bh_shares * first_price
    bh_final = bh_shares * last_price + bh_residual

    trading_days = max(0, len(portfolio) - 1)

    return {
        "ticker": t,
        "initial_cash": initial_cash,
        "dqn_final": round(float(dqn_final), 2),
        "bh_final": round(float(bh_final), 2),
        "dqn_return": round((float(dqn_final) / initial_cash - 1) * 100, 2),
        "bh_return": round((bh_final / initial_cash - 1) * 100, 2),
        "test_days": trading_days,
        "raw_rows": len(hist),
        "n_data": n_data,
        "period": EVAL_HISTORY_PERIOD,
        "train_test_split": TRAIN_SIZE,
        "simulation_days_requested": days_to_simulate,
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
    cash: Optional[float],
    shares: Optional[int],
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
    price_now = float(prices[idx])
    state = get_state(test_df[FEATURE_COLS], idx, c, vol)
    q_t, vol_logits_t, combined_idx, act, pred_volume = forward_policy(
        state, c, vol, price_now
    )

    q_np = q_t.cpu().numpy().copy()
    mask_np = get_legal_mask(c, vol, price_now).numpy()
    q_adj = q_np.copy()
    q_adj[~mask_np] = -1e30
    exp_q = np.exp(q_adj - np.max(q_adj))
    softmax_q = exp_q / exp_q.sum()
    confidence = float(softmax_q[combined_idx] * 100)

    vol_probs = F.softmax(vol_logits_t, dim=-1).cpu().numpy()
    action_names = ["HOLD", "BUY", "SELL"]

    q_breakdown = []
    for i, (a, sh) in enumerate(COMBINED_ACTIONS):
        q_breakdown.append({
            "index": i,
            "action": action_names[a],
            "shares": sh,
            "q": round(float(q_np[i]), 4),
            "legal": bool(mask_np[i]),
        })

    return {
        "ticker": ticker,
        "action": action_names[act],
        "action_index": act,
        "combined_index": combined_idx,
        "predicted_volume": pred_volume,
        "confidence": round(confidence, 1),
        "q_values": q_breakdown,
        "volume_logits": vol_logits_t.cpu().numpy().tolist(),
        "volume_probs": {str(model.volume_classes[i]): round(float(vol_probs[i]), 4) for i in range(len(model.volume_classes))},
        "price": round(float(prices[idx]), 2),
        "cash_used_in_state": c,
        "shares_held_in_state": vol,
    }


@app.get("/api/backtest/{ticker}")
async def get_backtest(ticker: str, initial_cash: float = 10000):
    try:
        return _execute_backtest(ticker, initial_cash)
    except ValueError as e:
        msg = str(e)
        if msg.startswith("No data for"):
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=400, detail=msg)


@app.get("/api/portfolio/summary")
async def get_portfolio_summary(initial_cash: float = 10000):
    results: dict = {}
    total_dqn = 0.0
    total_bh = 0.0
    capital_base = float(len(TICKERS) * initial_cash)

    for ticker in TICKERS:
        try:
            r = _execute_backtest(ticker, initial_cash)
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
        "period": EVAL_HISTORY_PERIOD,
        "train_test_split": TRAIN_SIZE,
        "simulation_rule": "max(240, n_test - 1 - SEQ_LEN) steps on held-out test split (same as test.py)",
        "total_dqn": round(total_dqn, 2),
        "total_bh": round(total_bh, 2),
        "total_dqn_return": round((total_dqn / capital_base - 1) * 100, 2) if capital_base else 0.0,
        "total_bh_return": round((total_bh / capital_base - 1) * 100, 2) if capital_base else 0.0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
