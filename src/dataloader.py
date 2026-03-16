import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_for_qlearning(history, scaler=None,
                             return_bins=10, 
                             vol_bins=10, 
                             ma_bins=10,
                             ma_short=5,
                             ma_long=20,
                             vol_window=10):
    
    

    history['log_return'] = np.log(history['Close'] / history['Close'].shift(1))
    

    history['ma_short'] = history['Close'].rolling(ma_short).mean()
    history['ma_long'] = history['Close'].rolling(ma_long).mean()
    

    history['ma_ratio'] = history['ma_short'] / history['ma_long']
    

    history['volatility'] = history['log_return'].rolling(vol_window).std()
    

    history = history.dropna().reset_index(drop=True)
    

    features = ['log_return', 'ma_ratio', 'volatility']
    if scaler is None:
        scaler = StandardScaler()
    history[features] = scaler.fit_transform(history[features])
    

    history['ret_bin'] = pd.qcut(history['log_return'], return_bins, labels=False)
    history['ma_bin'] = pd.qcut(history['ma_ratio'], ma_bins, labels=False)
    history['vol_bin'] = pd.qcut(history['volatility'], vol_bins, labels=False)
    

    history['state'] = (
        history['ret_bin'] * (ma_bins * vol_bins) +
        history['ma_bin'] * vol_bins +
        history['vol_bin']
    )
    
    n_states = return_bins * ma_bins * vol_bins
    
    return history, n_states, scaler


def train_test_split(df, train_size):
    train_index = int(len(df) * train_size)
    train_df = df.iloc[:train_index]
    test_df = df.iloc[train_index:]
    return train_df, test_df

def load_data(ticker, train_size, period="1y"):
    ticker = yf.Ticker(ticker)
    history = ticker.history(period=period)
    train_df, test_df = train_test_split(history, train_size)
    train_df, n_states, scaler = preprocess_for_qlearning(train_df, scaler=None)
    test_df, _, _ = preprocess_for_qlearning(test_df, scaler=scaler)
    
    return train_df, test_df, n_states, scaler



if __name__ == "__main__":
    ticker = "AAPL"
    train_size = 0.8
    period = "10y"
    train_df, test_df, n_states, scaler = load_data(ticker, train_size, period)
    
    print(len(train_df))
    print(len(test_df))
    
