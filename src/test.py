import pickle
import numpy as np
from Q_learning import step
from config import *
from dataloader import load_data

def test(test_df, days_window):
    with open('Q_table.pickle', 'rb') as f:
        Q_table = pickle.load(f)
    initial_cash = CASH
    initial_index = 0
    days_to_simulate = 240
    state = {
        "cash": initial_cash,
        "volume": 0,
        "price": test_df["Close"].iloc[initial_index],
        "time": test_df.index[initial_index]
    }
    reward_history = []
    baseline_value = initial_cash // test_df["Close"].iloc[initial_index] * test_df["Close"].iloc[initial_index+days_to_simulate-1]
    print(test_df.iloc[initial_index+days_to_simulate-1,:])

    for t in range(days_to_simulate):

        index = test_df.index.get_loc(state["time"])

        if index >= len(test_df) - 1:
            break
        
        market_state = test_df['state'].iloc[index]
        position = int(np.sign(state["volume"]))
        state_key = (market_state, position)

        if state_key in Q_table:
            action = np.argmax(Q_table[state_key])
        else:
            action = 0  # hold if unseen

        print(f"Day {t+1}: Action: {action}, "
            f"Cash: {state['cash']:.2f}, "
            f"Volume: {state['volume']}, "
            f"Price: {state['price']:.2f}")

        next_state, _ = step(test_df, state.copy(), action, index, days_window)
        reward_history.append(next_state["cash"] + next_state["volume"] * next_state["price"])

        state = next_state

    # Final portfolio value
    final_value = state["cash"] + state["volume"] * state["price"]
    print("Final portfolio value:", final_value)
    print("Baseline portfolio value:", baseline_value)
    print("Profit:", final_value - baseline_value)
    print("Profit percentage:", (final_value - baseline_value) / baseline_value * 100)

if __name__ == "__main__":
    ticker = "GOOG"
    period = "10y"

    _, test_df, _, _ = load_data(ticker, TRAIN_SIZE, period)
    test(test_df, DAYS_WINDOW)