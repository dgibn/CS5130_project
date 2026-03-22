import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import BINS


# ── technical indicator helpers ──────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line          # histogram


def _bollinger_width(series: pd.Series, window: int = 20) -> pd.Series:
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (2 * std) / sma                  # normalized band width


# ── feature engineering ──────────────────────────────────────────────

FEATURE_COLS = [
    "mom5",
    "ma_ratio",
    "volatility",
    "rsi",
    "macd_hist",
    "bb_width",
    "vol_ratio",
    "price_vs_sma50",
]

BIN_KEYS = [
    "mom",
    "ma",
    "vol",
    "rsi",
    "macd",
    "bb",
    "volr",
    "pvsma",
]


def _add_features(df: pd.DataFrame,
                  ma_short: int = 5,
                  ma_long: int = 20,
                  vol_window: int = 10,
                  rsi_period: int = 14,
                  mom_window: int = 5) -> pd.DataFrame:
    c, v = df["Close"], df["Volume"]

    # momentum
    df["mom5"] = c / c.shift(mom_window) - 1

    # moving-average ratio
    df["ma_ratio"] = c.rolling(ma_short).mean() / c.rolling(ma_long).mean()

    # realised volatility
    df["volatility"] = c.pct_change().rolling(vol_window).std()

    # RSI
    df["rsi"] = _rsi(c, period=rsi_period)

    # MACD histogram
    df["macd_hist"] = _macd(c)

    # Bollinger Band width (normalised)
    df["bb_width"] = _bollinger_width(c)

    # volume ratio  (today vs 20-day avg)
    df["vol_ratio"] = v / v.rolling(20).mean()

    # price relative to 50-day SMA
    df["price_vs_sma50"] = c / c.rolling(50).mean()

    return df


# ── discretisation ───────────────────────────────────────────────────

def _fit_bin_edges(df: pd.DataFrame, n_bins: int) -> dict[str, np.ndarray]:
    edges = {}
    for feat, key in zip(FEATURE_COLS, BIN_KEYS):
        _, e = pd.qcut(df[feat], n_bins, retbins=True,
                        labels=False, duplicates="drop")
        e[0], e[-1] = -np.inf, np.inf
        edges[key] = e
    return edges


def _apply_bins(df: pd.DataFrame, bin_edges: dict[str, np.ndarray]) -> pd.DataFrame:
    bin_cols = []
    for feat, key in zip(FEATURE_COLS, BIN_KEYS):
        col = f"{key}_bin"
        df[col] = pd.cut(df[feat], bins=bin_edges[key],
                          labels=False, include_lowest=True).astype(int)
        bin_cols.append(col)

    # encode composite state index (mixed-radix)
    n_feat = len(bin_cols)
    bases = [(len(bin_edges[k]) - 1) for k in BIN_KEYS]

    state = np.zeros(len(df), dtype=np.int64)
    for i, col in enumerate(bin_cols):
        multiplier = int(np.prod(bases[i + 1:])) if i < n_feat - 1 else 1
        state += df[col].values * multiplier

    df["state"] = state
    n_states = int(np.prod(bases))
    return df, n_states


# ── main preprocessing pipeline ─────────────────────────────────────

def preprocess_for_qlearning(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    bin_edges: dict = None,
    n_bins: int = BINS,
    **feature_kwargs,
) -> tuple[pd.DataFrame, int, StandardScaler, dict]:

    df = _add_features(df.copy(), **feature_kwargs)
    df = df.dropna().reset_index(drop=True)

    fit_scaler = scaler is None
    if fit_scaler:
        scaler = StandardScaler()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    else:
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    fit_bins = bin_edges is None
    if fit_bins:
        bin_edges = _fit_bin_edges(df, n_bins)

    df, n_states = _apply_bins(df, bin_edges)

    return df, n_states, scaler, bin_edges


# ── data loading & splitting ─────────────────────────────────────────

def train_test_split(df: pd.DataFrame, train_size: float):
    idx = int(len(df) * train_size)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def load_data(ticker: str, train_size: float = 0.8, period: str = "1y"):
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    train_raw, test_raw = train_test_split(hist, train_size)

    train_df, n_states, scaler, bin_edges = preprocess_for_qlearning(train_raw)
    test_df, _, _, _ = preprocess_for_qlearning(test_raw, scaler=scaler, bin_edges=bin_edges)

    return train_df, test_df, n_states, scaler


# ── quick sanity check ───────────────────────────────────────────────

if __name__ == "__main__":
    ticker = "GOOG"
    train_df, test_df, n_states, scaler = load_data(
        ticker, train_size=0.8, period="10y"
    )
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  States: {n_states}")
    print(f"Features: {FEATURE_COLS}")
    print(f"State range: [{train_df['state'].min()}, {train_df['state'].max()}]")
    print(f"\nSample row:\n{train_df[FEATURE_COLS + ['state']].iloc[-1]}")