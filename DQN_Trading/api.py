"""
FastAPI backend for the D3QN Trading Dashboard.
Loads the trained generalized model and serves live predictions.

Usage:
    pip install fastapi uvicorn
    python api.py
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import warnings

from env import TradingEnv
from agent import DQNAgent

warnings.filterwarnings("ignore")

app = FastAPI(title="D3QN Trading API")

# Allow any frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ──
TICKERS = ["GOOGL", "AAPL", "MSFT", "AMZN", "META"]
CHECKPOINT = "checkpoints/general_final.pt"
agent = None


# ──────────────────────────────────────────────
# STARTUP — load model once
# ──────────────────────────────────────────────
def load_agent():
    global agent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        state_dim=7,
        n_actions=3,
        hidden=128,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        eps_start=0.0,
        eps_end=0.0,
        eps_decay=1.0,
        buffer_size=1,
        batch_size=1,
        dueling=True,
        device=device,
    )
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            f"Train the model first: python general_model.py --episodes 1500"
        )
    agent.load(CHECKPOINT)
    agent.epsilon = 0.0  # pure greedy — no exploration
    print(f"  Agent loaded from {CHECKPOINT} (device: {device})")


@app.on_event("startup")
async def startup():
    print("Starting D3QN Trading API...")
    load_agent()
    print("API ready!")


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.get("/api/tickers")
async def get_tickers():
    """Return list of supported tickers."""
    return {"tickers": TICKERS}


@app.get("/api/price/{ticker}")
async def get_price(ticker: str, period: str = "6mo"):
    """
    Get price history for chart display.
    Returns current price, daily change %, and OHLCV history.
    """
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
async def get_prediction(ticker: str):
    """
    Get the D3QN agent's live trading signal for a ticker.
    Returns action (BUY/SELL/HOLD), confidence %, and raw Q-values.
    """
    ticker = ticker.upper()
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        hist.reset_index(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    # build env from latest data
    env = TradingEnv(hist, initial_cash=10_000, max_shares_per_trade=10)

    # get state at the latest available day
    state = env.reset(start_idx=env.n_steps - 1)

    # forward pass through the model
    with torch.no_grad():
        s = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        q_values = agent.policy_net(s).squeeze(0).cpu().numpy()

    action_idx = int(np.argmax(q_values))
    action_names = ["HOLD", "BUY", "SELL"]

    # softmax for confidence
    exp_q = np.exp(q_values - q_values.max())  # numerically stable
    softmax = exp_q / exp_q.sum()
    confidence = float(softmax[action_idx] * 100)

    return {
        "ticker": ticker,
        "action": action_names[action_idx],
        "confidence": round(confidence, 1),
        "q_values": {
            "hold": round(float(q_values[0]), 4),
            "buy": round(float(q_values[1]), 4),
            "sell": round(float(q_values[2]), 4),
        },
        "price": round(float(hist["Close"].iloc[-1]), 2),
    }


@app.get("/api/backtest/{ticker}")
async def get_backtest(ticker: str, initial_cash: float = 10000):
    """
    Run a full backtest on recent data using the trained model.
    Returns day-by-day portfolio values, actions, and comparison to buy & hold.
    Buy & Hold uses whole shares + residual cash.
    """
    ticker = ticker.upper()
    try:
        hist = yf.Ticker(ticker).history(period="2y")
        hist.reset_index(inplace=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch {ticker}: {e}")

    if hist.empty:
        raise HTTPException(status_code=404, detail=f"No data for {ticker}")

    # use last 20% as test period
    split = int(len(hist) * 0.8)
    test_df = hist.iloc[split:].reset_index(drop=True)
    env = TradingEnv(test_df, initial_cash=initial_cash, max_shares_per_trade=10)

    # run the agent through test data
    state = env.reset(start_idx=0)
    action_names = ["HOLD", "BUY", "SELL"]

    portfolio = [{
        "day": 0,
        "value": round(initial_cash, 2),
        "action": "START",
        "price": round(float(env.prices[0]), 2),
        "cash": round(initial_cash, 2),
        "shares": 0,
    }]

    while not env.done:
        action = agent.select_action(state, eval_mode=True)
        state, reward, done, info = env.step(action)
        portfolio.append({
            "day": len(portfolio),
            "value": round(info["portfolio_value"], 2),
            "action": action_names[action],
            "price": round(info["price"], 2),
            "cash": round(info["cash"], 2),
            "shares": info["shares"],
        })

    # buy & hold with whole shares + residual cash
    first_price = float(env.prices[0])
    last_price = float(env.prices[-1])
    bh_shares = int(initial_cash // first_price)
    bh_residual = initial_cash - bh_shares * first_price
    bh_final = bh_shares * last_price + bh_residual

    dqn_final = portfolio[-1]["value"]

    return {
        "ticker": ticker,
        "initial_cash": initial_cash,
        "dqn_final": round(dqn_final, 2),
        "bh_final": round(float(bh_final), 2),
        "dqn_return": round((dqn_final / initial_cash - 1) * 100, 2),
        "bh_return": round((bh_final / initial_cash - 1) * 100, 2),
        "test_days": len(portfolio),
        "portfolio": portfolio,
    }


@app.get("/api/simulate/{ticker}")
async def simulate_forward(ticker: str, days: int = 10, initial_cash: float = 10000, n_simulations: int = 50):
    """
    Monte Carlo forward simulation.
    Uses historical volatility to generate possible future price paths,
    then runs the D3QN agent on each path.
    Returns average, best, and worst case portfolio trajectories.
    """
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
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    last_price = float(close[-1])

    # generate simulated price paths
    all_paths = []
    for _ in range(n_simulations):
        prices = [last_price]
        for d in range(days):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1] * np.exp(shock))
        all_paths.append(prices)

    # run D3QN agent on each simulated path
    all_portfolios = []
    all_actions = []

    # we need historical data for indicators (MA, RSI, MACD need ~30+ days)
    # prepend real history before simulated future
    hist_tail = hist.tail(60).copy()  # last 60 days of real data

    for path in all_paths:
        cash = initial_cash
        shares = 0
        pv_list = [initial_cash]
        actions_list = []

        # build dataframe: real history + simulated future
        future_dates = pd.bdate_range(
            start=hist["Date"].iloc[-1] + pd.Timedelta(days=1),
            periods=days
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
            future_df
        ], ignore_index=True)

        try:
            env = TradingEnv(combined_df, initial_cash=initial_cash, max_shares_per_trade=10)

            # find where the future starts in the env
            # after indicator computation, some rows are dropped
            # we start the agent from where real data ends
            future_start = max(0, env.n_steps - days - 1)

            state = env.reset(start_idx=future_start)

            for d in range(min(days, env.n_steps - future_start - 1)):
                action = agent.select_action(state, eval_mode=True)
                state, reward, done, info = env.step(action)
                pv_list.append(info["portfolio_value"])
                actions_list.append(["HOLD", "BUY", "SELL"][action])
                if done:
                    break

            # pad if needed
            while len(pv_list) <= days:
                pv_list.append(pv_list[-1])
                actions_list.append("HOLD")

        except Exception:
            for d in range(days):
                pv_list.append(initial_cash)
                actions_list.append("HOLD")

        all_portfolios.append(pv_list[:days + 1])
        all_actions.append(actions_list[:days])

    # compute statistics
    portfolios_arr = np.array(all_portfolios)
    avg_pv = portfolios_arr.mean(axis=0).tolist()
    best_pv = portfolios_arr.max(axis=0).tolist()
    worst_pv = portfolios_arr.min(axis=0).tolist()
    p25_pv = np.percentile(portfolios_arr, 25, axis=0).tolist()
    p75_pv = np.percentile(portfolios_arr, 75, axis=0).tolist()

    # most common action per day
    from collections import Counter
    common_actions = []
    for d in range(days):
        day_actions = [all_actions[s][d] for s in range(n_simulations) if d < len(all_actions[s])]
        if day_actions:
            common_actions.append(Counter(day_actions).most_common(1)[0][0])
        else:
            common_actions.append("HOLD")

    # avg price paths
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
    """
    Run backtest on all tickers with $2k each and return combined summary.
    This calls the backtest endpoint internally for each ticker.
    """
    results = {}
    total_dqn = 0
    total_bh = 0

    for ticker in TICKERS:
        try:
            hist = yf.Ticker(ticker).history(period="2y")
            hist.reset_index(inplace=True)
            split = int(len(hist) * 0.8)
            test_df = hist.iloc[split:].reset_index(drop=True)
            env = TradingEnv(test_df, initial_cash=2000, max_shares_per_trade=10)

            state = env.reset(start_idx=0)
            while not env.done:
                action = agent.select_action(state, eval_mode=True)
                state, _, done, info = env.step(action)

            dqn_final = info["portfolio_value"]

            # buy & hold
            first_price = float(env.prices[0])
            last_price = float(env.prices[-1])
            bh_shares = int(2000 // first_price)
            bh_residual = 2000 - bh_shares * first_price
            bh_final = bh_shares * last_price + bh_residual

            results[ticker] = {
                "dqn": round(dqn_final, 2),
                "bh": round(float(bh_final), 2),
                "dqn_ret": round((dqn_final / 2000 - 1) * 100, 2),
                "bh_ret": round((bh_final / 2000 - 1) * 100, 2),
            }
            total_dqn += dqn_final
            total_bh += bh_final

        except Exception as e:
            results[ticker] = {
                "dqn": 2000, "bh": 2000,
                "dqn_ret": 0, "bh_ret": 0,
                "error": str(e),
            }

    return {
        "tickers": results,
        "total_dqn": round(total_dqn, 2),
        "total_bh": round(total_bh, 2),
        "total_dqn_return": round((total_dqn / 10000 - 1) * 100, 2),
        "total_bh_return": round((total_bh / 10000 - 1) * 100, 2),
    }


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)