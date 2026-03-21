import pickle
import numpy as np
import torch
from Q_learning import step
from config import *
from dataloader import load_data, train_test_split, FEATURE_COLS
from dataloader_rnn import preprocess_for_rnn
from RNN import RNNQNetwork
import yfinance as yf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_qlearning(test_df, days_window):
    with open("Q_table.pickle", "rb") as f:
        Q_table = pickle.load(f)

    initial_cash = CASH
    days_to_simulate = min(240, len(test_df) - 1)
    state = {
        "cash": initial_cash,
        "volume": 0,
        "price": test_df["Close"].iloc[0],
        "time": test_df.index[0],
    }
    baseline_value = (
        initial_cash // test_df["Close"].iloc[0]
        * test_df["Close"].iloc[days_to_simulate - 1]
    )

    for t in range(days_to_simulate):
        index = test_df.index.get_loc(state["time"])
        if index >= len(test_df) - 1:
            break

        market_state = test_df["state"].iloc[index]
        position = int(np.sign(state["volume"]))
        state_key = (market_state, position)

        action = np.argmax(Q_table[state_key]) if state_key in Q_table else 0

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


def test_rnn_single(ticker):
    """Evaluate the trained RNN-DQN on one ticker's test set."""
    hist = yf.Ticker(ticker).history(period=PERIOD)
    _, test_raw = train_test_split(hist, TRAIN_SIZE)
    test_df, _ = preprocess_for_rnn(test_raw)

    features = torch.tensor(
        test_df[FEATURE_COLS].values, dtype=torch.float32, device=device,
    )
    prices = test_df["Close"].values
    n_data = len(test_df)

    days_to_simulate = min(240, n_data - SEQ_LEN - 1)
    cash, volume = float(CASH), 0
    start_idx = SEQ_LEN
    baseline_value = (
        cash // prices[start_idx]
        * prices[start_idx + days_to_simulate - 1]
    )

    for t in range(days_to_simulate):
        idx = start_idx + t
        if idx >= n_data - 1:
            break

        state_seq = features[idx - SEQ_LEN + 1 : idx + 1]
        with torch.no_grad():
            q = model(state_seq.unsqueeze(0))
            action = q[0, -1, :].argmax().item()

        price_now = prices[idx]

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


if __name__ == "__main__":
    if MODE == "qlearning":
        ticker = TICKERS[0]
        _, test_df, _, _ = load_data(ticker, TRAIN_SIZE, PERIOD)
        test_qlearning(test_df, DAYS_WINDOW)

    elif MODE == "rnn":
        print(f"Using device: {device}")
        model = RNNQNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load("rnn_q_network.pt", map_location=device))
        model.eval()

        for ticker in TICKERS:
            print(f"\n{'='*20} {ticker} {'='*20}")
            test_rnn_single(ticker)

    else:
        raise ValueError(f"Unknown MODE: {MODE}")
