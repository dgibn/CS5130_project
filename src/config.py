# ── mode ──────────────────────────────────────────────────────────────
MODE = "rnn"            # "qlearning" or "rnn"

# ── shared ────────────────────────────────────────────────────────────
TICKER = "GOOG"
PERIOD = "10y"
DAYS_WINDOW = 5
CASH = 10000
VOLUME = 0
PRICE = None
TIME = None
HISTORY = None
ACTIONS = [0, 1, 2]           # 0: hold, 1: buy, 2: sell
ACTIONS1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NUM_EPISODES = 1000
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.05
TRANSACTION_COST = 0.001
TRAIN_SIZE = 0.8

# ── Q-learning specific ──────────────────────────────────────────────
BINS = 4
ALPHA_MAX = 0.1
ALPHA_MIN = 0.01
DECAY_RATE = 0.999

# ── RNN-DQN specific ─────────────────────────────────────────────────
INPUT_DIM = 8                 # must match len(FEATURE_COLS) in dataloader
HIDDEN_DIM = 64
OUTPUT_DIM = len(ACTIONS)     # 3
SEQ_LEN = DAYS_WINDOW
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
TARGET_UPDATE_FREQ = 10       # sync target network every N episodes
REPLAY_BUFFER_SIZE = 10000
