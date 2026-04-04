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
from DQN import DuelingDQN
import yfinance as yf

yf.set_tz_cache_location("/tmp") 

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


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


def test_rnn_single(ticker):
    """Evaluate the trained RNN-DQN on one ticker's test set."""
    hist = yf.Ticker(ticker).history(period=PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)

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

    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break

        state_seq = features[idx - SEQ_LEN + 1 : idx + 1]
        with torch.no_grad():
            q = model(state_seq.unsqueeze(0))
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

def baseline_buy_and_hold(ticker, cash):
    """Buy & hold: spend $cash on ticker at entry, track daily portfolio value."""
    hist = yf.Ticker(ticker).history(period=PERIOD)
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


def test_Qnet_single(q_net,ticker,cash,volume):
    """Evaluate the trained QNet on one ticker's test set."""

    # ── Load & preprocess ────────────────────────────────────────────────
    hist = yf.Ticker(ticker).history(period=PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)

    prices = test_df["Close"].values
    n_data = len(test_df)

    # ── Simulation setup ─────────────────────────────────────────────────
    start_idx        = SEQ_LEN
    days_to_simulate = max(240, n_data - 1 - SEQ_LEN)


    # ── Baseline (buy & hold) ───────────────────────────────────────────
    entry_price    = prices[start_idx]
    shares_bh      = int(cash // entry_price)
    baseline_value = (
        cash // prices[start_idx]
        * prices[start_idx + days_to_simulate - 1]
    )
    bh_leftover    = cash - shares_bh * entry_price
    baseline_value = bh_leftover + shares_bh * prices[start_idx + days_to_simulate - 1]
    
    cash_history = []

    def get_state(df, idx: int, cash: float, volume: int) -> torch.Tensor:
        row = df.iloc[idx - SEQ_LEN + 1 : idx + 1].values.flatten().astype(np.float32)
        portfolio = np.array([
            cash / CASH,           # normalised cash
            float(volume) / 10.0,  # normalised volume
        ], dtype=np.float32)
        return torch.tensor(np.concatenate([row, portfolio]), dtype=torch.float32)

    q_net.eval()

    # ── Simulation loop ───────────────────────────────────────────────────
    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break

        state = get_state(test_df[FEATURE_COLS], idx, cash, volume).to(device)

        with torch.no_grad():
            # BUG 3 FIX: reset noise before every inference forward pass
            q_net.reset_noise()
            action, vol = q_net(state.unsqueeze(0))
            action = action.argmax().item()
            vol= vol.argmax().item()
            
            VOL_CLASSES     = [1, 5, 10, 15, 20] 
            vol = VOL_CLASSES[vol]
        
        price_now = prices[idx]
        print(f"Day {t+1}: TESTING Action: {action}, Volume: {vol}, Cash: {cash:>10.2f}, TotalVolume: {volume:>4}, Price: {price_now:>8.2f}")
            

        if action == 1 and cash >= price_now:    # buy
            shares   = min(int(cash // price_now), vol)
            cash    -= shares * price_now
            volume  += shares
            position = 1
        elif action == 2 and volume > 0:         # sell
            shares   = min(volume, vol)
            cash    += shares * price_now
            volume  -= shares
            position = 0 if volume == 0 else position

        cash_history.append(cash + volume * price_now)
    # ── Final report ──────────────────────────────────────────────────────
    final_price = prices[min(start_idx + days_to_simulate - 1, n_data - 1)]
    final_value = cash + volume * final_price

    # print(f"\n{'─'*40}")
    # print(f"Final portfolio value  : {final_value:>10.2f}")
    # print(f"Baseline (buy & hold)  : {baseline_value:>10.2f}")
    # print(f"Profit over baseline   : {final_value - baseline_value:>10.2f}")

    return final_value,baseline_value, cash_history

if __name__ == "__main__":

    print(f"Using device: {device}")
    model = DuelingDQN(162, len(ACTIONS)).to(device)
    model.load_state_dict(torch.load("Q_net.pt", map_location=device))
    model.eval()
    total_val = 0.0 
    Baseline_val = 0.0
    all_histories = []
    for ticker in TICKERS:
        val1, val2, cash_history = test_Qnet_single(model, ticker, float(CASH), 0)
        print(val1, val2)
        total_val += val1
        Baseline_val += val2
        all_histories.append(cash_history)
    total_cash_history = np.sum(all_histories, axis=0)
    csv_path = 'cash_history.csv'
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Method"] + [f"Day{i+1}" for i in range(len(total_cash_history))]
            writer.writerow(header)
        writer.writerow(["duelingDQN"] + list(total_cash_history))
    print(f"\n{'─'*40}")
    print(f"Final portfolio value  : {total_val:>10.2f}")
    print(f"Baseline (buy & hold)  : {Baseline_val:>10.2f}")
    print(f"Profit over baseline   : {(total_val-Baseline_val):>10.2f}")
    '''
   csv_path = 'cash_history.csv'
    all_baseline_histories = []
    for ticker in TICKERS:
        bh = baseline_buy_and_hold(ticker, float(CASH))
        all_baseline_histories.append(bh)
    total_baseline_history = np.sum(all_baseline_histories, axis=0)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["baseline"] + list(total_baseline_history))'''



    
    exit()
    if MODE == "qlearning":
        total_baseline = 0
        total_final = 0

        csv_path = 'results.csv'
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

        all_histories = []
        for ticker in TICKERS:
            _, test_df, _, _ = load_data(ticker, TRAIN_SIZE, PERIOD)
            baseline, final, cash_history = test_qlearning(test_df, DAYS_WINDOW)
            total_baseline += baseline
            total_final += final
            all_histories.append(cash_history)

        total_cash_history = np.sum(all_histories, axis=0)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["Method"] + [f"Day{i+1}" for i in range(len(total_cash_history))]
                writer.writerow(header)
            writer.writerow([MODE] + list(total_cash_history))

        print(f"\nFinal portfolio value:    {total_final:.2f}")
        print(f"Baseline (buy & hold):   {total_baseline:.2f}")
        print(f"Profit over baseline:    {total_final - total_baseline:.2f}")

    elif MODE == "rnn":
        print(f"Using device: {device}")
        model = RNNQNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load("rnn_q_network.pt", map_location=device))
        model.eval()

        total_baseline = 0
        total_final = 0

        all_histories = []
        for ticker in TICKERS:
            print(f"\n{'='*20} {ticker} {'='*20}")
            baseline, final, cash_history = test_rnn_single(ticker)
            total_baseline += baseline
            total_final += final
            all_histories.append(cash_history)

        total_cash_history = np.sum(all_histories, axis=0)

        csv_path = 'cash_history.csv'
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["Method"] + [f"Day{i+1}" for i in range(len(total_cash_history))]
                writer.writerow(header)
            writer.writerow([MODE] + list(total_cash_history))

        print(f"\nFinal portfolio value:    {total_final:.2f}")
        print(f"Baseline (buy & hold):   {total_baseline:.2f}")
        print(f"Profit over baseline:    {total_final - total_baseline:.2f}")
    
    # else:
    #     raise ValueError(f"Unknown MODE: {MODE}")
