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
from Q_network import DuelingDQN
from loss import double_dqn_loss , dqn_loss
import yfinance as yf
from dataloader import load_data
from Q_learning import Q_learning

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def train_qlearning():
    
    

    ticker = TICKERS[0]
    train_df, test_df, n_states, scaler = load_data(ticker, TRAIN_SIZE, PERIOD)
    Q_table = Q_learning(
        train_df, DAYS_WINDOW, ACTIONS, ACTIONS1,
        NUM_EPISODES_QL, GAMMA, EPSILON, DECAY_RATE,
    )

    with open("Q_table.pickle", "wb") as f:
        pickle.dump(Q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved Q_table.pickle")


def train_rnn():
    print(f"Using device: {device}")

    # --- load all tickers, fit one shared scaler ---------------------------
    all_features = []
    all_prices = []
    all_test_features = []
    all_test_prices = []
    scaler = None

    for ticker in TICKERS:
        print(f"Loading {ticker}...")
        hist = yf.Ticker(ticker).history(period=PERIOD)
        train_raw, test_raw = train_test_split(hist, TRAIN_SIZE)
        train_df, scaler = preprocess_for_rnn(train_raw, scaler=scaler)
        test_df, _ = preprocess_for_rnn(test_raw, scaler=scaler)
        all_features.append(
            torch.tensor(train_df[FEATURE_COLS].values, dtype=torch.float32)
        )
        all_prices.append(train_df["Close"].values)
        all_test_features.append(
            torch.tensor(test_df[FEATURE_COLS].values, dtype=torch.float32)
        )
        all_test_prices.append(test_df["Close"].values)

    # --- networks & optimizer ----------------------------------------------
    online_net = RNNQNetwork(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
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

    def eval_test_raw_return():
        online_net.eval()
        all_returns = []
        with torch.no_grad():
            for features, prices in zip(all_test_features, all_test_prices):
                n_data = len(features)
                if n_data <= SEQ_LEN:
                    continue
                cash, volume = float(CASH), 0
                start_cash = cash

                for idx in range(SEQ_LEN, n_data):
                    state_seq = features[idx - SEQ_LEN + 1 : idx + 1]
                    q = online_net(state_seq.unsqueeze(0).to(device))
                    action = q[0, -1, :].argmax().item()
                    price_now = prices[idx]

                    if action == 1:
                        shares = min(int(cash // price_now), 10)
                        cash -= shares * price_now
                        volume += shares
                    elif action == 2:
                        shares = min(volume, 10)
                        cash += shares * price_now
                        volume -= shares

                final_value = cash + volume * prices[-1]
                all_returns.append((final_value - start_cash) / start_cash)
        online_net.train()
        if not all_returns:
            return float("-inf")
        return float(np.mean(all_returns))

    # --- training loop -----------------------------------------------------
    epsilon = EPSILON
    best_test_raw_return = float("-inf")

    for episode in tqdm(range(NUM_EPISODES), desc="RNN-DQN"):
        ti = random.randint(0, len(TICKERS) - 1)
        features = all_features[ti]
        prices = all_prices[ti]
        n_data = len(features)

        idx = random.randint(SEQ_LEN, n_data - 2)
        cash, volume = float(CASH), 0
        steps = 0

        while idx < n_data - 1 and steps < MAX_STEPS_PER_EPISODE:
            state_seq = features[idx - SEQ_LEN + 1 : idx + 1]   # (L, C)
            price_now = prices[idx]
            price_next = prices[idx + 1]

            # epsilon-greedy using Q-values from the last timestep
            if random.random() < epsilon:
                action = random.choice(ACTIONS)
            else:
                with torch.no_grad():
                    q = online_net(state_seq.unsqueeze(0).to(device))
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
            buffer.append((state_seq, action, r, next_seq, float(done)))
            steps += 1

            # train on a mini-batch every N steps
            if len(buffer) >= BATCH_SIZE and steps % TRAIN_EVERY == 0:
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

        test_raw_return = eval_test_raw_return()
        if test_raw_return > best_test_raw_return:
            best_test_raw_return = test_raw_return
            torch.save(online_net.state_dict(), "rnn_q_network.pt")
            print(f"Episode {episode + 1}: new best test raw return = {best_test_raw_return:.4f}")

    print(f"Saved best rnn_q_network.pt (test raw return: {best_test_raw_return:.4f})")



def train_qNET():
    # ── Pre-load all tickers ──────────────────────────────────────────────
    all_train_dfs = {}
    scaler = None
    for ticker in TICKERS:
        hist = yf.Ticker(ticker).history(period=PERIOD)
        train_raw, _ = train_test_split(hist, TRAIN_SIZE)
        train_df, scaler = preprocess_for_rnn(train_raw, scaler=scaler)
        all_train_dfs[ticker] = train_df

    q_net      = DuelingDQN(15, len(ACTIONS)).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()

    optimizer = torch.optim.RMSprop(q_net.parameters(), lr=2.5e-4, eps=0.01, alpha=0.95)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100_000, gamma=0.5)

    WARMUP_STEPS = max(BATCH_SIZE * 4, 1_000)
    buffer       = deque(maxlen=REPLAY_BUFFER_SIZE)
    total_loss, steps = 0.0, 0

    def epsilon_schedule(episode, eps_start=1.0, eps_end=0.01, decay=0.995):
        return max(eps_end, eps_start * (decay ** episode))

    def compute_reward(df, idx, action, position):
        price_now  = df["Close"].iloc[idx]
        price_next = df["Close"].iloc[min(idx + 1, len(df) - 1)]
        ret        = (price_next - price_now) / price_now
        if action == 1:
            position = 1
        elif action == 2:
            position = 0
        return float(position * ret), position

    def get_state(df, idx):
        return torch.tensor(
            df.iloc[idx].values.flatten().astype(np.float32), dtype=torch.float32
        )

    def sample_buffer(batch_size):
        batch = random.sample(buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s).to(device),
            torch.tensor(a, dtype=torch.long,    device=device),
            torch.tensor(r, dtype=torch.float32, device=device),
            torch.stack(s2).to(device),
            torch.tensor(d, dtype=torch.float32, device=device),
        )

    for episode in tqdm(range(NUM_EPISODES), desc="Dueling DQN"):

        epsilon = epsilon_schedule(episode)

        # ── BUG 3 FIX: loop over ALL tickers each episode ────────────────
        for ticker, train_df in all_train_dfs.items():
            position = 0
            T        = len(train_df) - 1

            for t in range(T):
                state = get_state(train_df, t)

                if random.random() < epsilon:
                    action = random.randrange(len(ACTIONS))
                else:
                    q_net.eval()
                    with torch.no_grad():
                        # ── BUG 4 FIX: reset noise before every forward pass
                        q_net.reset_noise()
                        s_t    = state.unsqueeze(0).to(device)
                        action = q_net(s_t).argmax(dim=1).item()
                    q_net.train()

                reward, position = compute_reward(train_df, t, action, position)
                next_state       = get_state(train_df, t + 1)
                done             = (t == T - 1)
                buffer.append([state, action, reward, next_state, done])

        if len(buffer) < WARMUP_STEPS:
            continue

        states, actions, rewards, next_states, dones = sample_buffer(BATCH_SIZE)

        q_net.reset_noise()
        target_net.reset_noise()

        q_values      = q_net(states)
        next_q_online = q_net(next_states)
        with torch.no_grad():
            next_q_target = target_net(next_states)

        # ── zero_grad only once, right before backward ────────
        optimizer.zero_grad()
        loss = dqn_loss(q_values, actions, rewards, next_q_online, next_q_target, dones, GAMMA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps      += 1

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Avg Loss: {total_loss/steps:.4f}")
            total_loss, steps = 0.0, 0

    torch.save(q_net.state_dict(), "q_net.pt")
    print("Saved q_net.pt")


if __name__ == "__main__":
    train_qNET()
    # if MODE == "qlearning":
    #     train_qlearning()
    # elif MODE == "rnn":
    #     train_rnn()
    
    # # elif MODE == "qnet":
        
    # else:
    #     raise ValueError(f"Unknown MODE: {MODE}")
