from config import *
import pickle
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import os
from collections import deque
from tqdm import tqdm
from dataloader import train_test_split, FEATURE_COLS
from dataloader_rnn import preprocess_for_rnn
from RNN import RNNQNetwork
from DQN import DuelingDQN
from loss import double_dqn_loss, dqn_loss
import yfinance as yf
import yfinance.cache as yfc
from dataloader import load_data
from Q_learning import Q_learning

yf.set_tz_cache_location("/tmp")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

PERIOD = "10y"

# ── Flattened action × volume space ──────────────────────────────────────────
VOL_CLASSES = [1, 5, 10, 15, 20]

COMBINED_ACTIONS = [(0, 0)]                        # hold
for v in VOL_CLASSES:
    COMBINED_ACTIONS.append((1, v))                # buy v shares
    COMBINED_ACTIONS.append((2, v))                # sell v shares
NUM_COMBINED = len(COMBINED_ACTIONS)               # 11



def train_qNET():

    # Pre-load all tickers
    all_train_dfs = {}
    scaler        = None
    for ticker in TICKERS:
        hist              = yf.Ticker(ticker).history(period=PERIOD)
        train_raw, _      = train_test_split(hist, TRAIN_SIZE)
        train_df, scaler  = preprocess_for_rnn(train_raw, scaler=scaler)
        all_train_dfs[ticker] = train_df

    # ── Single-headed Dueling DQN: 11 combined actions ──────────────────────
    STATE_DIM = 162
    q_net      = DuelingDQN(STATE_DIM, NUM_COMBINED).to(device)
    target_net = copy.deepcopy(q_net)
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    WARMUP_STEPS       = max(BATCH_SIZE * 4, 1_000)
    TRAIN_STEPS_PER_EP = 4
    buffer             = deque(maxlen=REPLAY_BUFFER_SIZE)
    total_loss         = 0.0
    steps              = 0
    best_test_return   = float("-inf")

    REWARD_SCALE = 50.0

    # ── Boltzmann (softmax) temperature schedule ────────────────────────
    TEMP_START = 2.0          # high temperature  → near-uniform exploration
    TEMP_END   = 0.1          # low temperature   → near-greedy
    TEMP_DECAY_FRAC = 0.85    # decay over 85% of training

    def temperature_schedule(episode, total_eps=NUM_EPISODES_QL):
        frac = min(1.0, episode / (TEMP_DECAY_FRAC * total_eps))
        return TEMP_START + frac * (TEMP_END - TEMP_START)

    def get_state(df, idx: int, cash: float, volume: int) -> torch.Tensor:
        row       = df.iloc[idx - SEQ_LEN + 1 : idx + 1].values.flatten().astype(np.float32)
        portfolio = np.array([cash / CASH, float(volume) / 20.0], dtype=np.float32)
        return torch.tensor(np.concatenate([row, portfolio]), dtype=torch.float32)

    def get_legal_mask(cash: float, volume: int, price: float) -> torch.Tensor:
        """Returns a boolean mask of shape [NUM_COMBINED]. True = legal."""
        mask = torch.ones(NUM_COMBINED, dtype=torch.bool)
        for i, (act, vol) in enumerate(COMBINED_ACTIONS):
            if act == 1 and cash < price * vol:
                mask[i] = False          # can't afford this buy
            elif act == 2 and volume < vol:
                mask[i] = False          # don't own enough to sell
        return mask

    def compute_reward(df, idx, action, shares, pre_volume, pre_cash):
        price_now  = df["Close"].iloc[idx]
        price_next = df["Close"].iloc[min(idx + 1, len(df) - 1)]
        log_ret    = float(np.log(price_next / price_now))

        if action == 1:
            reward  = log_ret * REWARD_SCALE * shares
            reward -= TRANSACTION_COST
            return float(reward)

        if action == 2:
            reward  = -log_ret * REWARD_SCALE * shares
            reward -= TRANSACTION_COST
            return float(reward)

        # action == 0: hold
        if pre_volume > 0:
            return float(log_ret * pre_volume * REWARD_SCALE)
        else:
            return -5.0

    def sample_buffer(batch_size):
        batch = random.sample(buffer, batch_size)
        s, a, r, s2, d, m = zip(*batch)
        return (
            torch.stack(s).to(device),
            torch.tensor(a,  dtype=torch.long,    device=device),
            torch.tensor(r,  dtype=torch.float32, device=device),
            torch.stack(s2).to(device),
            torch.tensor(d,  dtype=torch.float32, device=device),
            torch.stack(m).to(device),                              # next-state legal masks
        )

    # ── Pre-load test data for return-based checkpointing ──────────────
    all_test_dfs = {}
    for ticker in TICKERS:
        hist          = yf.Ticker(ticker).history(period=PERIOD)
        _, test_raw   = train_test_split(hist, TRAIN_SIZE)
        test_df, _    = preprocess_for_rnn(test_raw, scaler=scaler)
        all_test_dfs[ticker] = test_df

    def eval_test_return():
        """Run greedy policy on every ticker's test set, return mean portfolio return."""
        q_net.eval()
        all_returns = []
        for ticker, tdf in all_test_dfs.items():
            prices = tdf["Close"].values
            n      = len(tdf)
            if n <= SEQ_LEN + 1:
                continue
            cash, vol = float(CASH), 0
            for t in range(SEQ_LEN, n - 1):
                st = get_state(tdf[FEATURE_COLS], t, cash, vol).to(device)
                price_now = prices[t]
                mask = get_legal_mask(cash, vol, price_now)
                with torch.no_grad():
                    q_net.reset_noise()
                    out    = q_net(st.unsqueeze(0))
                    q_vals = out[0] if isinstance(out, tuple) else out
                    q_vals[0, ~mask.to(device)] = float("-inf")
                    ai = q_vals.argmax(dim=1).item()
                act, ps = COMBINED_ACTIONS[ai]
                if act == 1:
                    s = min(ps, int(cash // price_now))
                    cash -= s * price_now; vol += s
                elif act == 2:
                    s = min(ps, vol)
                    cash += s * price_now; vol -= s
            final = cash + vol * prices[-1]
            all_returns.append((final - CASH) / CASH)
        q_net.train()
        return float(np.mean(all_returns)) if all_returns else float("-inf")

    for episode in tqdm(range(NUM_EPISODES_QL), desc="Dueling DQN"):
        temp     = temperature_schedule(episode)
        ti       = random.randint(0, len(TICKERS) - 1)
        train_df = all_train_dfs[TICKERS[ti]]
        cash     = float(CASH)
        volume   = 0
        T        = len(train_df) - 1
        idx      = random.randint(SEQ_LEN, T - 1)

        for t in range(idx, T):
            state     = get_state(train_df[FEATURE_COLS], t, cash, volume)
            price_now = train_df["Close"].iloc[t]
            mask      = get_legal_mask(cash, volume, price_now)

            # ── Boltzmann (softmax) action selection ───────────────────────
            q_net.eval()
            with torch.no_grad():
                q_net.reset_noise()
                out    = q_net(state.unsqueeze(0).to(device))
                q_vals = out[0] if isinstance(out, tuple) else out    # [1, 11]
                q_vals[0, ~mask.to(device)] = float("-inf")

                # softmax over legal actions with temperature
                probs      = F.softmax(q_vals[0] / temp, dim=-1)
                action_idx = torch.multinomial(probs, 1).item()
            q_net.train()

            action, pred_shares = COMBINED_ACTIONS[action_idx]

            # ── Execute action ──────────────────────────────────────────────
            pre_cash   = cash
            pre_volume = volume
            shares     = 0

            if action == 1:
                shares  = min(pred_shares, int(cash // price_now))
                cash   -= shares * price_now
                volume += shares
            elif action == 2:
                shares  = min(pred_shares, volume)
                cash   += shares * price_now
                volume -= shares

            # ── Reward ──────────────────────────────────────────────────────
            reward     = compute_reward(train_df, t, action, shares,
                                        pre_volume, pre_cash)
            done       = (t == T - 1)
            next_state = get_state(train_df[FEATURE_COLS], t + 1, cash, volume)

            # Store next-state legal mask for masked Double-DQN target
            next_price = train_df["Close"].iloc[min(t + 1, T)]
            next_mask  = get_legal_mask(cash, volume, next_price)

            buffer.append([state, action_idx, reward, next_state, done, next_mask])

        if len(buffer) < WARMUP_STEPS:
            continue

        # ── Multiple training steps per episode ─────────────────────────────
        for _ in range(TRAIN_STEPS_PER_EP):
            states, actions, rewards, next_states, dones, next_masks = sample_buffer(BATCH_SIZE)

            # Clip rewards to prevent TD target explosion
            rewards = rewards.clamp(-10.0, 10.0)

            q_net.reset_noise()
            target_net.reset_noise()

            q_out      = q_net(states)
            nq_online  = q_net(next_states)
            with torch.no_grad():
                nq_target = target_net(next_states)

            q_values      = q_out[0]      if isinstance(q_out,      tuple) else q_out
            next_q_online = nq_online[0]  if isinstance(nq_online,  tuple) else nq_online
            next_q_target = nq_target[0]  if isinstance(nq_target,  tuple) else nq_target

            # Masked Double DQN: pick best *legal* next action via online net
            masked_next_online = next_q_online.clone()
            masked_next_online[~next_masks] = float("-inf")

            td_loss = dqn_loss(
                q_values, actions, rewards,
                masked_next_online, next_q_target,
                dones, GAMMA,
            )

            optimizer.zero_grad()
            td_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += td_loss.item()
            steps      += 1

        scheduler.step()

        if (episode + 1) % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (episode + 1) % 10 == 0:
            avg_loss = total_loss / max(steps, 1)
            print(f"Episode {episode + 1} | TD Loss: {avg_loss:.4f} | Temp: {temp:.3f}")
            total_loss, steps = 0.0, 0

        # Evaluate on test set every 100 episodes and checkpoint best
        if (episode + 1) % 100 == 0:
            test_ret = eval_test_return()
            print(f"  >> Test return: {test_ret:.4f}  (best: {best_test_return:.4f})")
            if test_ret > best_test_return:
                best_test_return = test_ret
                torch.save(q_net.state_dict(),
                           f"Q_net_best{episode+1}.pt")
                print(f"  >> New best checkpoint saved!")

    torch.save(q_net.state_dict(), "Q_net_final.pt")
    print(f"Saved Q_net_final.pt  (last episode)")



if __name__ == "__main__":
    train_qNET()