import random
from tqdm import tqdm
import numpy as np

def reward(prev_state, next_state, action):
    if prev_state["volume"] == 0 and action == 2:
        return -10  # Penalty for selling without position
    if prev_state["cash"] < prev_state["price"] and action == 1:
        return -10  # Penalty for buying without cash
    prev_value = prev_state["cash"] + prev_state["volume"] * prev_state["price"]
    next_value = next_state["cash"] + next_state["volume"] * next_state["price"]
    return next_value - prev_value


def step(df, state, action, index, days_window):
    price = df["Close"].iloc[index]
    revenue = 0
    if action == 0: # hold
        revenue = 0
    elif action == 1: # buy
        shares_to_buy = min(state["cash"] // price, 20) # can only buy up to 20 shares
        cost = shares_to_buy * price
        state["cash"] -= cost
        state["volume"] += shares_to_buy
        revenue = -cost
    elif action == 2: # sell
        shares_to_sell = min(state["volume"], 20) # can only sell up to 20 shares
        revenue = shares_to_sell * price
        state["cash"] += revenue
        state["volume"] -= shares_to_sell
    state = {
        "cash": state["cash"],
        "volume": state["volume"],
        "price": df["Close"].iloc[index+1], # next price
        "time": df.index[index+1],
        "history": df['state'].iloc[index-days_window+1:index+1].tolist() # last (days_window) days of closing prices
    }
    return state, revenue



def Q_learning(train_df, days_window, actions, actions1, num_episodes, gamma, epsilon, decay_rate):
    

    Q_table = {}
    update_counts = {}

    for episode in tqdm(range(num_episodes)):

        initial_index = random.randint(days_window, len(train_df)-1)

        state = {
            "cash": 10000,
            "volume": 0,
            "price": train_df["Close"].iloc[initial_index],
            "time": train_df.index[initial_index]
        }

        done = False

        while not done:

            index = train_df.index.get_loc(state["time"])

            if index >= len(train_df) - 1:
                break

            market_state = train_df['state'].iloc[index]
            position = int(np.sign(state["volume"]))
            state_key = (market_state, position)

            if state_key not in Q_table:
                Q_table[state_key] = np.zeros(len(actions))
                update_counts[state_key] = np.zeros(len(actions))

            if np.random.rand() < epsilon:
                action = np.random.choice(actions)
            else:
                action = np.argmax(Q_table[state_key])


            next_state, _ = step(train_df, state.copy(), action, index, days_window)

            
            reward_value = reward(state, next_state, action)

            next_index = train_df.index.get_loc(next_state["time"])
            
            next_market_state = train_df['state'].iloc[next_index]
            next_position = int(np.sign(next_state["volume"]))
            next_state_key = (next_market_state, next_position)

            if next_state_key not in Q_table:
                Q_table[next_state_key] = np.zeros(len(actions))
                update_counts[next_state_key] = np.zeros(len(actions))


            update_counts[state_key][action] += 1
            eta = 1.0 / update_counts[state_key][action]


            Q_table[state_key][action] += eta * (
                reward_value
                + gamma * np.max(Q_table[next_state_key])
                - Q_table[state_key][action]
            )

            state = next_state

        epsilon *= decay_rate

    return Q_table


