import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from dataloader import _add_features, FEATURE_COLS, train_test_split
from config import DAYS_WINDOW
import yfinance as yf


def preprocess_for_rnn(df, scaler=None):
    """Feature engineering + scaling, no discretisation."""
    df = _add_features(df.copy())
    df = df.dropna().reset_index(drop=True)

    if scaler is None:
        scaler = StandardScaler()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    else:
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])

    return df, scaler


class TradingSequenceDataset(Dataset):
    """Sliding-window dataset that yields (feature_seq, price_now, price_next)."""

    def __init__(self, features: np.ndarray, prices: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.prices = torch.tensor(prices, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        seq = self.features[idx : idx + self.seq_len]           # (L, C)
        price_now  = self.prices[idx + self.seq_len - 1]
        price_next = self.prices[idx + self.seq_len]
        return seq, price_now, price_next


def load_data(ticker: str, train_size: float = 0.8, period: str = "1y",
              seq_len: int = DAYS_WINDOW):
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    train_raw, test_raw = train_test_split(hist, train_size)

    train_df, scaler = preprocess_for_rnn(train_raw)
    test_df, _       = preprocess_for_rnn(test_raw, scaler=scaler)

    train_ds = TradingSequenceDataset(
        train_df[FEATURE_COLS].values, train_df["Close"].values, seq_len)
    test_ds = TradingSequenceDataset(
        test_df[FEATURE_COLS].values, test_df["Close"].values, seq_len)

    return train_ds, test_ds, scaler


if __name__ == "__main__":
    train_ds, test_ds, scaler = load_data("GOOG", train_size=0.8, period="10y")
    seq, p_now, p_next = train_ds[0]
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")
    print(f"Sequence shape: {seq.shape}  (L={seq.shape[0]}, C={seq.shape[1]})")
    print(f"Price now: {p_now:.2f}  Price next: {p_next:.2f}")
