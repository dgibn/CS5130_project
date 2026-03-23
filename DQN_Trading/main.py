"""
Deep Q-Network Stock Trading Agent — Entry Point
=================================================
Downloads AAPL data, trains the DQN agent, evaluates on held-out
test data, and produces a summary plot.

Usage
-----
    python main.py                     
    python main.py --eval-only         
    python main.py --episodes 1000     

Project structure
-----------------
    main.py            
    env.py             ← TradingEnv
    model.py           ← Dueling DQN network
    agent.py           ← DQNAgent (policy + target net, ε-greedy, learn)
    replay_buffer.py   ← experience replay
    train.py           ← training loop
    evaluate.py        ← backtest + metrics + plots
    requirements.txt   ← pip dependencies
"""

import argparse
import warnings

import torch
import yfinance as yf

from env import TradingEnv
from agent import DQNAgent
from train import train
from evaluate import backtest, compute_metrics, plot_results

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="DQN Stock Trading Agent")
    p.add_argument("--ticker",    type=str,   default="AAPL",  help="Yahoo Finance ticker symbol")
    p.add_argument("--period",    type=str,   default="10y",   help="Data download period")
    p.add_argument("--episodes",  type=int,   default=500,     help="Training episodes")
    p.add_argument("--hidden",    type=int,   default=128,     help="Hidden layer size")
    p.add_argument("--lr",        type=float, default=1e-3,    help="Learning rate")
    p.add_argument("--gamma",     type=float, default=0.99,    help="Discount factor")
    p.add_argument("--batch",     type=int,   default=64,      help="Batch size")
    p.add_argument("--buffer",    type=int,   default=50_000,  help="Replay buffer capacity")
    p.add_argument("--eval-only", action="store_true",         help="Skip training, just evaluate")
    p.add_argument("--checkpoint",type=str,   default="checkpoints/dqn_final.pt", help="Checkpoint path")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Downloading {args.ticker} data ({args.period})...")
    ticker = yf.Ticker(args.ticker)
    history = ticker.history(period=args.period)
    history.reset_index(inplace=True)
    print(f"  Total rows: {len(history)}")

    split = int(len(history) * 0.8)
    train_df = history.iloc[:split].reset_index(drop=True)
    test_df  = history.iloc[split:].reset_index(drop=True)

    train_env = TradingEnv(train_df, initial_cash=10_000, max_shares_per_trade=10)
    test_env  = TradingEnv(test_df,  initial_cash=10_000, max_shares_per_trade=10)

    print(f"  State dim  : {train_env.state_dim}")
    print(f"  Train steps: {train_env.n_steps}")
    print(f"  Test steps : {test_env.n_steps}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(
        state_dim=train_env.state_dim,
        n_actions=TradingEnv.N_ACTIONS,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        tau=0.005,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        buffer_size=args.buffer,
        batch_size=args.batch,
        dueling=True,
        device=device,
    )
    print(f"  Device     : {device}")

    if args.eval_only:
        agent.load(args.checkpoint)
        train_stats = None
    else:
        print(f"\n{'='*50}")
        print(f"  TRAINING — {args.episodes} episodes")
        print(f"{'='*50}")
        train_stats = train(
            agent, train_env,
            n_episodes=args.episodes,
            verbose=True,
            save_every=100,
        )

    print(f"\n{'='*50}")
    print("  BACKTESTING ON TEST DATA")
    print(f"{'='*50}")

    test_pv, test_actions = backtest(agent, test_env, start_idx=0)
    metrics = compute_metrics(test_pv, initial_cash=10_000)

    buy_hold_return = (test_env.prices[-1] / test_env.prices[0] - 1) * 100

    print(f"  DQN Final Value    : ${metrics['final_value']:,.2f}")
    print(f"  DQN Total Return   : {metrics['total_return_pct']:+.2f}%")
    print(f"  DQN Sharpe Ratio   : {metrics['sharpe_ratio']:.3f}")
    print(f"  DQN Max Drawdown   : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Buy&Hold Return    : {buy_hold_return:+.2f}%")

    if train_stats is not None:
        plot_results(
            train_stats=train_stats,
            test_pv=test_pv,
            test_actions=test_actions,
            test_prices=test_env.prices,
            save_path="results/dqn_results.png",
        )


if __name__ == "__main__":
    main()