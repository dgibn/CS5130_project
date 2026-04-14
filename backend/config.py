# ── mode (train.py: qlearning | rnn | qnet) ───────────────────────────
# test.py also accepts qnet_volume (11-action DQN from train_volume.py).
#   qlearning    — tabular Q; needs Q_table.pickle
#   rnn          — RNNQNetwork; needs rnn_q_network.pt
#   qnet         — 3-action Q_network.DuelingDQN; e.g. Q_net4.pt from train_qNET
#   qnet_volume  — 11-action DQN.DuelingDQN; e.g. Q_net_best.pt from train_volume.py
MODE = "qnet_volume"

# ── shared ────────────────────────────────────────────────────────────
TICKERS = ["GOOG","AAPL","MSFT", "AMZN", "META"]
PERIOD = "10y"
# yfinance window for API backtest, test.py, and offline eval — matches /api/backtest (training may still use PERIOD).
EVAL_HISTORY_PERIOD = "10y"
DAYS_WINDOW = 5
CASH = 10000
VOLUME = 0
PRICE = None
TIME = None
HISTORY = None
ACTIONS = [0, 1, 2]           # 0: hold, 1: buy, 2: sell
ACTIONS1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
TRANSACTION_COST = 0.001
TRAIN_SIZE = 0.8

# ── Q-learning specific ──────────────────────────────────────────────
NUM_EPISODES_QL = 2000
BINS = 4
ALPHA_MAX = 0.1
ALPHA_MIN = 0.01
DECAY_RATE = 0.999

# ── RNN-DQN specific ─────────────────────────────────────────────────
INPUT_DIM = 8                 # must match len(FEATURE_COLS) in dataloader
HIDDEN_DIM = 128
OUTPUT_DIM = len(ACTIONS)     # 3
NUM_LAYERS = 4
SEQ_LEN = 20
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
NUM_EPISODES = 3000
TARGET_UPDATE_FREQ = 20       # sync target network every N episodes
REPLAY_BUFFER_SIZE = 50000
TRAIN_EVERY = 4               # gradient update every N env steps
MAX_STEPS_PER_EPISODE = 200   # cap episode length