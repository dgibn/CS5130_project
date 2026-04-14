import pickle
import numpy as np
import torch
import os
import csv
from Q_learning import step
from config import *
from dataloader import load_data, train_test_split, FEATURE_COLS
from dataloader_rnn import preprocess_for_rnn
from RNN import RNNQNetwork
from DQN import DuelingDQN as DuelingDQNVolume
from Q_network import DuelingDQN as DuelingDQN3
import yfinance as yf

yf.set_tz_cache_location("/tmp")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Must match train.py ─────────────────────────────────────────────────────
VOL_CLASSES = [1, 5, 10, 15, 20]

COMBINED_ACTIONS = [(0, 0)]
for v in VOL_CLASSES:
    COMBINED_ACTIONS.append((1, v))
    COMBINED_ACTIONS.append((2, v))
NUM_COMBINED = len(COMBINED_ACTIONS)

# Checkpoints (defaults match train.py / train_volume.py; override via env)
QNET_CHECKPOINT = os.environ.get("QNET_CHECKPOINT", "Q_net4.pt")
DQN_VOL_CHECKPOINT = os.environ.get("DQN_CHECKPOINT", "Q_net_best.pt")
RNN_CHECKPOINT = os.environ.get("RNN_CHECKPOINT", "rnn_q_network.pt")


def _fit_rnn_scaler_from_train():
    """Match train_rnn: fit scaler on first ticker, transform rest on train splits."""
    scaler = None
    for t in TICKERS:
        hist = yf.Ticker(t).history(period=PERIOD)
        train_raw, _ = train_test_split(hist, TRAIN_SIZE)
        _, scaler = preprocess_for_rnn(train_raw, scaler=scaler)
    return scaler


def append_cash_history_csv(method, total_cash_history):
    total_cash_history = np.asarray(total_cash_history)
    csv_path = "cash_history.csv"
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Method"] + [f"Day{i+1}" for i in range(len(total_cash_history))]
            writer.writerow(header)
        writer.writerow([method] + list(total_cash_history))


# ── Q-Learning test ──────────────────────────────────────────────────────────

def test_qlearning(test_df, days_window):
    with open("Q_table.pickle", "rb") as f:
        Q_table = pickle.load(f)

    cash_history = []
    initial_cash = CASH
    days_to_simulate = max(240, len(test_df) - 1 - SEQ_LEN)
    state = {
        "cash": initial_cash,
        "volume": 0,
        "price": test_df["Close"].iloc[0],
        "time": test_df.index[0],
    }
    bh_shares = int(initial_cash // test_df["Close"].iloc[0])
    bh_leftover = initial_cash - bh_shares * test_df["Close"].iloc[0]
    baseline_value = bh_leftover + bh_shares * test_df["Close"].iloc[days_to_simulate - 1]

    for t in range(days_to_simulate):
        index = test_df.index.get_loc(state["time"])
        if index >= len(test_df) - 1:
            break

        market_state = test_df["state"].iloc[index]
        position = int(np.sign(state["volume"]))
        state_key = (market_state, position)

        action = np.argmax(Q_table[state_key]) if state_key in Q_table else 0

        cash_history.append(state["cash"] + state["volume"] * state["price"])

        print(
            f"Day {t+1}: Action: {action}, "
            f"Cash: {state['cash']:.2f}, "
            f"Volume: {state['volume']}, "
            f"Price: {state['price']:.2f}"
        )

        next_state, _ = step(test_df, state.copy(), action, index, days_window)
        state = next_state

    final_value = state["cash"] + state["volume"] * state["price"]
    print(f"\nFinal portfolio value:    {final_value:.2f}")
    print(f"Baseline (buy & hold):   {baseline_value:.2f}")
    print(f"Profit over baseline:    {final_value - baseline_value:.2f}")

    return baseline_value, final_value, cash_history


# ── RNN-DQN test ─────────────────────────────────────────────────────────────

def test_rnn_single(ticker, model, scaler):
    """Evaluate the trained RNN-DQN on one ticker's test set (scaler from train_rnn)."""
    hist = yf.Ticker(ticker).history(period=PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw, scaler=scaler)

    cash_history = []
    features = torch.tensor(
        test_df[FEATURE_COLS].values, dtype=torch.float32, device=device,
    )
    prices = test_df["Close"].values
    n_data = len(test_df)

    start_idx        = SEQ_LEN
    days_to_simulate = max(240, n_data - 1 - SEQ_LEN)
    cash, volume = float(CASH), 0

    bh_shares = int(cash // prices[start_idx])
    bh_leftover = cash - bh_shares * prices[start_idx]
    baseline_value = bh_leftover + bh_shares * prices[start_idx + days_to_simulate - 1]

    model.eval()
    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break

        state_seq = features[idx - SEQ_LEN + 1 : idx + 1]
        with torch.no_grad():
            q = model(state_seq.unsqueeze(0).to(device))
            action = q[0, -1, :].argmax().item()

        price_now = prices[idx]

        cash_history.append(cash + volume * price_now)
        print(
            f"Day {t+1}: Action: {action}, "
            f"Cash: {cash:.2f}, "
            f"Volume: {volume}, "
            f"Price: {price_now:.2f}"
        )

        if action == 1:
            shares = min(int(cash // price_now), 10)
            cash -= shares * price_now
            volume += shares
        elif action == 2:
            shares = min(volume, 10)
            cash += shares * price_now
            volume -= shares

    final_price = prices[min(start_idx + days_to_simulate - 1, n_data - 1)]
    final_value = cash + volume * final_price
    print(f"\nFinal portfolio value:    {final_value:.2f}")
    print(f"Baseline (buy & hold):   {baseline_value:.2f}")
    print(f"Profit over baseline:    {final_value - baseline_value:.2f}")

    return baseline_value, final_value, cash_history


# ── 3-action Dueling DQN (train.py train_qNET) ─────────────────────────────

def test_qnet_three_action(q_net, ticker, cash, volume):
    """Evaluate Q_network.DuelingDQN (3 actions, max 10 shares) on one ticker."""
    hist = yf.Ticker(ticker).history(period=EVAL_HISTORY_PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)

    prices = test_df["Close"].values
    n_data = len(test_df)

    start_idx        = SEQ_LEN
    days_to_simulate = max(240, n_data - 1 - SEQ_LEN)

    entry_price    = prices[start_idx]
    shares_bh      = int(cash // entry_price)
    bh_leftover    = cash - shares_bh * entry_price
    baseline_value = bh_leftover + shares_bh * prices[start_idx + days_to_simulate - 1]

    cash_history = []

    def get_state(df, idx: int, cash: float, volume: int) -> torch.Tensor:
        row = df.iloc[idx - SEQ_LEN + 1 : idx + 1].values.flatten().astype(np.float32)
        portfolio = np.array([
            cash / CASH,
            float(volume) / 10.0,
        ], dtype=np.float32)
        return torch.tensor(np.concatenate([row, portfolio]), dtype=torch.float32)

    q_net.eval()
    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break

        state     = get_state(test_df[FEATURE_COLS], idx, cash, volume).to(device)
        price_now = prices[idx]

        with torch.no_grad():
            q_net.reset_noise()
            out = q_net(state.unsqueeze(0))
            q_vals = out if not isinstance(out, tuple) else out[0]
            action = q_vals.argmax(dim=1).item()

        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]
        print(
            f"Day {t+1}: {action_name:>4} | "
            f"Cash: {cash:>10.2f} | Volume: {volume:>4} | "
            f"Price: {price_now:>8.2f}"
        )

        if action == 1 and cash >= price_now:
            shares = min(int(cash // price_now), 10)
            cash -= shares * price_now
            volume += shares
        elif action == 2 and volume > 0:
            shares = min(volume, 10)
            cash += shares * price_now
            volume -= shares

        cash_history.append(cash + volume * price_now)

    final_price = prices[min(start_idx + days_to_simulate - 1, n_data - 1)]
    final_value = cash + volume * final_price

    print(f"\nFinal portfolio value:    {final_value:.2f}")
    print(f"Baseline (buy & hold):   {baseline_value:.2f}")
    print(f"Profit over baseline:    {final_value - baseline_value:.2f}")

    return final_value, baseline_value, cash_history


# ── Baseline buy & hold ──────────────────────────────────────────────────────

def baseline_buy_and_hold(ticker, cash):
    """Buy & hold: spend $cash on ticker at entry, track daily portfolio value."""
    hist = yf.Ticker(ticker).history(period=EVAL_HISTORY_PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)

    prices = test_df["Close"].values
    n_data = len(test_df)

    start_idx = SEQ_LEN
    days_to_simulate = max(240, n_data - 1 - SEQ_LEN)

    entry_price = prices[start_idx]
    shares = int(cash // entry_price)
    leftover = cash - shares * entry_price

    cash_history = []
    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break
        cash_history.append(leftover + shares * prices[idx])

    return cash_history


# ── Dueling DQN test (flattened action × volume) ────────────────────────────

def test_Qnet_single(q_net, ticker, cash, volume):
    """Evaluate the trained QNet with flattened action×volume on one ticker."""

    hist = yf.Ticker(ticker).history(period=EVAL_HISTORY_PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)

    prices = test_df["Close"].values
    n_data = len(test_df)

    start_idx        = SEQ_LEN
    days_to_simulate = max(240, n_data - 1 - SEQ_LEN)

    # Baseline (buy & hold)
    entry_price    = prices[start_idx]
    shares_bh      = int(cash // entry_price)
    bh_leftover    = cash - shares_bh * entry_price
    baseline_value = bh_leftover + shares_bh * prices[start_idx + days_to_simulate - 1]

    cash_history = []

    def get_state(df, idx: int, cash: float, volume: int) -> torch.Tensor:
        row = df.iloc[idx - SEQ_LEN + 1 : idx + 1].values.flatten().astype(np.float32)
        portfolio = np.array([
            cash / CASH,
            float(volume) / 20.0,
        ], dtype=np.float32)
        return torch.tensor(np.concatenate([row, portfolio]), dtype=torch.float32)

    def get_legal_mask(cash, volume, price):
        mask = torch.ones(NUM_COMBINED, dtype=torch.bool)
        for i, (act, vol) in enumerate(COMBINED_ACTIONS):
            if act == 1 and cash < price * vol:
                mask[i] = False
            elif act == 2 and volume < vol:
                mask[i] = False
        return mask

    q_net.eval()

    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break

        state     = get_state(test_df[FEATURE_COLS], idx, cash, volume).to(device)
        price_now = prices[idx]
        mask      = get_legal_mask(cash, volume, price_now)

        with torch.no_grad():
            q_net.reset_noise()
            out    = q_net(state.unsqueeze(0))
            q_vals = out[0] if isinstance(out, tuple) else out        # [1, 11]
            q_vals[0, ~mask.to(device)] = float("-inf")
            action_idx = q_vals.argmax(dim=1).item()

        action, pred_shares = COMBINED_ACTIONS[action_idx]

        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]
        print(
            f"Day {t+1}: {action_name:>4} x{pred_shares:<2} | "
            f"Cash: {cash:>10.2f} | Volume: {volume:>4} | "
            f"Price: {price_now:>8.2f}"
        )

        if action == 1 and cash >= price_now:
            shares  = min(pred_shares, int(cash // price_now))
            cash   -= shares * price_now
            volume += shares
        elif action == 2 and volume > 0:
            shares  = min(pred_shares, volume)
            cash   += shares * price_now
            volume -= shares

        cash_history.append(cash + volume * price_now)

    final_price = prices[min(start_idx + days_to_simulate - 1, n_data - 1)]
    final_value = cash + volume * final_price

    return final_value, baseline_value, cash_history


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(f"Using device: {device}")

    total_val = 0.0
    Baseline_val = 0.0
    all_histories = []

    if MODE == "qlearning":
        # Uses config PERIOD; train.py train_qlearning shadows PERIOD with "5y" — set config if needed.
        _, test_df, _, _ = load_data(TICKERS[0], TRAIN_SIZE, PERIOD)
        baseline_val, final_val, cash_history = test_qlearning(test_df, DAYS_WINDOW)
        Baseline_val = baseline_val
        total_val = final_val
        append_cash_history_csv(MODE, cash_history)

    elif MODE == "rnn":
        rnn_model = RNNQNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
        rnn_model.load_state_dict(torch.load(RNN_CHECKPOINT, map_location=device))
        rnn_model.eval()
        rnn_scaler = _fit_rnn_scaler_from_train()
        for ticker in TICKERS:
            b, f, h = test_rnn_single(ticker, rnn_model, rnn_scaler)
            total_val += f
            Baseline_val += b
            all_histories.append(h)
        total_cash_history = np.sum(all_histories, axis=0)
        append_cash_history_csv(MODE, total_cash_history)

    elif MODE == "qnet":
        qnet = DuelingDQN3(162, len(ACTIONS)).to(device)
        qnet.load_state_dict(torch.load(QNET_CHECKPOINT, map_location=device))
        qnet.eval()
        for ticker in TICKERS:
            val1, val2, cash_history = test_qnet_three_action(
                qnet, ticker, float(CASH), 0
            )
            print(val1, val2)
            total_val += val1
            Baseline_val += val2
            all_histories.append(cash_history)
        total_cash_history = np.sum(all_histories, axis=0)
        append_cash_history_csv(MODE, total_cash_history)

    elif MODE == "qnet_volume":
        vol_model = DuelingDQNVolume(162, NUM_COMBINED).to(device)
        vol_model.load_state_dict(torch.load(DQN_VOL_CHECKPOINT, map_location=device))
        vol_model.eval()
        for ticker in TICKERS:
            val1, val2, cash_history = test_Qnet_single(
                vol_model, ticker, float(CASH), 0
            )
            print(val1, val2)
            total_val += val1
            Baseline_val += val2
            all_histories.append(cash_history)
        total_cash_history = np.sum(all_histories, axis=0)
        append_cash_history_csv(MODE, total_cash_history)

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print(f"\n{'─'*40}")
    print(f"Final portfolio value  : {total_val:>10.2f}")
    print(f"Baseline (buy & hold)  : {Baseline_val:>10.2f}")
    print(f"Profit over baseline   : {(total_val - Baseline_val):>10.2f}")