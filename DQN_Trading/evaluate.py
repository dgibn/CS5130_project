"""
Evaluation (backtesting) and plotting utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent import DQNAgent
from env import TradingEnv


def backtest(agent: DQNAgent, env: TradingEnv, start_idx: int = 0):
    """
    Run a full episode with the greedy policy (no exploration).

    Returns
    -------
    portfolio_values : list[float]
    actions_taken    : list[int]
    """
    state = env.reset(start_idx=start_idx)
    portfolio_values = [env._portfolio_value(env.prices[env.idx])]
    actions_taken = []

    while not env.done:
        action = agent.select_action(state, eval_mode=True)
        state, _, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        actions_taken.append(action)

    return portfolio_values, actions_taken


def compute_metrics(portfolio_values, initial_cash: float = 10_000):
    pv = np.array(portfolio_values)
    total_return = (pv[-1] / initial_cash - 1) * 100
    daily_returns = np.diff(pv) / pv[:-1]
    sharpe = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    )
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / (peak + 1e-8)
    max_dd = drawdown.max() * 100
    return {
        "total_return_pct": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "final_value": pv[-1],
    }



def plot_results(
    train_stats: dict,
    test_pv: list[float],
    test_actions: list[int],
    test_prices: np.ndarray,
    initial_cash: float = 10_000,
    save_path: str = "results/dqn_results.png",
):
    """
    Generate a 2×2 summary plot and save it.
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # — Training reward —
    rewards = train_stats["rewards"]
    ax = axes[0, 0]
    ax.plot(rewards, alpha=0.3, label="Episode reward")
    ax.plot(pd.Series(rewards).rolling(20).mean(), label="20-ep MA", linewidth=2)
    ax.set_title("Training: Episode Rewards")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(True)

    losses = train_stats["losses"]
    ax = axes[0, 1]
    ax.plot(losses, alpha=0.3, label="Avg loss")
    ax.plot(pd.Series(losses).rolling(20).mean(), label="20-ep MA", linewidth=2)
    ax.set_title("Training: Huber Loss")
    ax.set_xlabel("Episode")
    ax.legend()
    ax.grid(True)

    buy_hold = initial_cash / test_prices[0] * test_prices
    ax = axes[1, 0]
    ax.plot(test_pv, label="DQN Agent")
    ax.plot(buy_hold, label="Buy & Hold", linestyle="--")
    ax.axhline(initial_cash, color="gray", linestyle=":", label="Initial Cash")
    ax.set_title("Test: Portfolio Value")
    ax.set_xlabel("Day")
    ax.set_ylabel("$")
    ax.legend()
    ax.grid(True)

    labels = ["Hold", "Buy", "Sell"]
    counts = [test_actions.count(i) for i in range(3)]
    colors = ["#888888", "#2ecc71", "#e74c3c"]
    ax = axes[1, 1]
    ax.bar(labels, counts, color=colors)
    ax.set_title("Test: Action Distribution")
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved → {save_path}")