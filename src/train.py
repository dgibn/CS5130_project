from config import *
import pickle
import copy
import random
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from dataloader import train_test_split, FEATURE_COLS
from dataloader_rnn import preprocess_for_rnn
from RNN import RNNQNetwork
from loss import double_dqn_loss
import yfinance as yf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_qlearning():
    from dataloader import load_data
    from Q_learning import Q_learning

    train_df, test_df, n_states, scaler = load_data(TICKER, TRAIN_SIZE, PERIOD)
    Q_table = Q_learning(
        train_df, DAYS_WINDOW, ACTIONS, ACTIONS1,
        NUM_EPISODES, GAMMA, EPSILON, DECAY_RATE,
    )

    with open("Q_table.pickle", "wb") as f:
        pickle.dump(Q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved Q_table.pickle")


def train_rnn():
    print(f"Using device: {device}")

    # --- data --------------------------------------------------------------
    hist = yf.Ticker(TICKER).history(period=PERIOD)
    train_raw, _ = train_test_split(hist, TRAIN_SIZE)
    train_df, scaler = preprocess_for_rnn(train_raw)

    features = torch.tensor(
        train_df[FEATURE_COLS].values, dtype=torch.float32, device=device,
    )
    prices = train_df["Close"].values
    n_data = len(train_df)

    # --- networks & optimizer ----------------------------------------------
    online_net = RNNQNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    target_net = copy.deepcopy(online_net)
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

    # --- replay buffer -----------------------------------------------------
    buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def sample_buffer(batch_size):
        batch = random.sample(buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s).to(device),
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.stack(s2).to(device),
            torch.tensor(d, dtype=torch.float32, device=device),
        )

    # --- training loop -----------------------------------------------------
    epsilon = EPSILON

    for episode in tqdm(range(NUM_EPISODES), desc="RNN-DQN"):
        idx = random.randint(SEQ_LEN, n_data - 2)
        cash, volume = float(CASH), 0

        while idx < n_data - 1:
            state_seq = features[idx - SEQ_LEN + 1 : idx + 1]   # (L, C)
            price_now = prices[idx]
            price_next = prices[idx + 1]

            # epsilon-greedy using Q-values from the last timestep
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                with torch.no_grad():
                    q = online_net(state_seq.unsqueeze(0))       # (1, L, A)
                    action = q[0, -1, :].argmax().item()

            # reward (mirrors Q_learning.py)
            if volume == 0 and action == 2:
                r = -10.0
            elif cash < price_now and action == 1:
                r = -10.0
            else:
                r = 0.0
                if volume > 0 and price_now > 0:
                    r = float(np.log(price_next / price_now)) * volume
                if action in (1, 2):
                    r -= TRANSACTION_COST

            # execute action
            if action == 1:
                shares = min(int(cash // price_now), 10)
                cash -= shares * price_now
                volume += shares
            elif action == 2:
                shares = min(volume, 10)
                cash += shares * price_now
                volume -= shares

            idx += 1
            done = idx >= n_data - 1

            next_seq = (
                features[idx - SEQ_LEN + 1 : idx + 1]
                if not done
                else state_seq
            )
            buffer.append((state_seq.cpu(), action, r, next_seq.cpu(), float(done)))

            # train on a mini-batch
            if len(buffer) >= BATCH_SIZE:
                s, a, rew, s2, d = sample_buffer(BATCH_SIZE)
                q_vals = online_net(s)[:, -1, :]                 # (B, A)
                with torch.no_grad():
                    nq_online = online_net(s2)[:, -1, :]
                    nq_target = target_net(s2)[:, -1, :]
                loss = double_dqn_loss(
                    q_vals, a, rew, nq_online, nq_target, d, GAMMA,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), 1.0)
                optimizer.step()

        epsilon = max(EPSILON_MIN, 1.0 - episode / (0.7 * NUM_EPISODES))

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

    torch.save(online_net.state_dict(), "rnn_q_network.pt")
    print("Saved rnn_q_network.pt")


if __name__ == "__main__":
    if MODE == "qlearning":
        train_qlearning()
    elif MODE == "rnn":
        train_rnn()
    else:
        raise ValueError(f"Unknown MODE: {MODE}")
