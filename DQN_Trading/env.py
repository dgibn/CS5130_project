"""
Trading Environment for the DQN agent.

Wraps historical price data into a gym-like step/reset interface.
Computes technical indicators and normalises them as state features.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TradingEnv:
    """
    Actions
    -------
    0 = Hold
    1 = Buy  (up to max_shares_per_trade shares)
    2 = Sell (up to max_shares_per_trade shares)

    State vector (all normalised)
    -----------------------------
    log_return, ma_ratio, volatility, rsi, macd,
    position_ratio, cash_ratio
    """

    HOLD, BUY, SELL = 0, 1, 2
    N_ACTIONS = 3

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 10_000,
        max_shares_per_trade: int = 10,
    ):
        self.raw = df.copy()
        self.initial_cash = initial_cash
        self.max_shares = max_shares_per_trade

        self._build_features()
        # +2 for position_ratio and cash_ratio
        self.state_dim = self.features.shape[1] + 2

    # ------------------------------------------------------------------ #
    #  Feature engineering                                                 #
    # ------------------------------------------------------------------ #
    def _build_features(self):
        df = self.raw.copy()
        close = df["Close"]

        df["log_ret"]  = np.log(close / close.shift(1))
        df["ma5"]      = close.rolling(5).mean()
        df["ma20"]     = close.rolling(20).mean()
        df["ma_ratio"] = df["ma5"] / df["ma20"]
        df["vol10"]    = df["log_ret"].rolling(10).std()
        df["rsi"]      = self._rsi(close, 14)
        df["macd"]     = self._macd(close)

        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        feature_cols = ["log_ret", "ma_ratio", "vol10", "rsi", "macd"]
        self.scaler = StandardScaler()
        self.features = pd.DataFrame(
            self.scaler.fit_transform(df[feature_cols]),
            columns=feature_cols,
        )
        self.prices = df["Close"].values
        self.n_steps = len(self.prices)

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line - signal_line

    # ------------------------------------------------------------------ #
    #  Gym-like interface                                                  #
    # ------------------------------------------------------------------ #
    def reset(self, start_idx: int | None = None) -> np.ndarray:
        """Reset the environment and return the initial state."""
        if start_idx is None:
            self.idx = 0
        else:
            self.idx = max(0, min(start_idx, self.n_steps - 2))

        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
        return self._get_state()

    def step(self, action: int):
        """
        Execute *action*, advance one day.

        Returns
        -------
        next_state : np.ndarray
        reward     : float   (change in portfolio value / initial_cash)
        done       : bool
        info       : dict
        """
        assert not self.done, "Episode finished — call reset()."

        price = self.prices[self.idx]
        prev_value = self._portfolio_value(price)

        # ── execute trade ──
        if action == self.BUY:
            max_buy = int(self.cash // price)
            n_buy = min(max_buy, self.max_shares)
            self.cash -= n_buy * price
            self.shares += n_buy

        elif action == self.SELL:
            n_sell = min(self.shares, self.max_shares)
            self.cash += n_sell * price
            self.shares -= n_sell

        # advance one day
        self.idx += 1
        new_price = self.prices[self.idx]
        new_value = self._portfolio_value(new_price)

        # reward = normalised PnL
        reward = (new_value - prev_value) / self.initial_cash

        # small penalty for impossible / wasteful actions
        if action == self.BUY and self.cash < price:
            reward -= 0.01
        if action == self.SELL and self.shares == 0:
            reward -= 0.01

        if self.idx >= self.n_steps - 1:
            self.done = True

        info = {
            "portfolio_value": new_value,
            "cash": self.cash,
            "shares": self.shares,
            "price": new_price,
        }
        return self._get_state(), reward, self.done, info

    def _portfolio_value(self, price: float) -> float:
        return self.cash + self.shares * price

    def _get_state(self) -> np.ndarray:
        market = self.features.iloc[self.idx].values.astype(np.float32)
        price = self.prices[self.idx]
        pv = self._portfolio_value(price)
        pos_ratio  = (self.shares * price) / (pv + 1e-8)
        cash_ratio = self.cash / (pv + 1e-8)
        return np.concatenate([market, [pos_ratio, cash_ratio]])