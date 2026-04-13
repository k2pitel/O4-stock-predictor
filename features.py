"""
features.py — Feature engineering module.

Creates technical indicators, returns, volatility, lagged features and targets
from raw OHLCV data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series,
                 fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": histogram,
    })


def compute_bollinger(series: pd.Series, period: int = 20,
                      num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands (upper, middle, lower)."""
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return pd.DataFrame({
        "BB_Upper": mid + num_std * std,
        "BB_Mid": mid,
        "BB_Lower": mid - num_std * std,
    })


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   vix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build feature matrix from raw OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [Open, High, Low, Close, Volume].
    vix : pd.DataFrame | None
        Optional VIX data; its Close will be added as a feature.

    Returns
    -------
    pd.DataFrame
        Feature matrix with the original index (NaN rows dropped).
    """
    feat = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    close = feat["Close"]

    # Moving averages
    for w in [10, 20, 50]:
        feat[f"MA_{w}"] = close.rolling(window=w).mean()

    # RSI
    feat["RSI"] = compute_rsi(close)

    # MACD
    macd_df = compute_macd(close)
    feat = pd.concat([feat, macd_df], axis=1)

    # Bollinger Bands
    bb_df = compute_bollinger(close)
    feat = pd.concat([feat, bb_df], axis=1)

    # Returns
    feat["Return_1d"] = close.pct_change()
    feat["Return_5d"] = close.pct_change(5)

    # Momentum
    feat["Momentum_1d"] = close.diff(1)
    feat["Momentum_5d"] = close.diff(5)

    # Volatility (rolling std of returns)
    feat["Volatility_10d"] = feat["Return_1d"].rolling(10).std()
    feat["Volatility_20d"] = feat["Return_1d"].rolling(20).std()

    # Lagged close prices
    for lag in [1, 2, 3, 5]:
        feat[f"Close_Lag{lag}"] = close.shift(lag)

    # VIX
    if vix is not None:
        vix_close = vix["Close"].reindex(feat.index).ffill()
        feat["VIX"] = vix_close

    feat.dropna(inplace=True)
    return feat


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def add_classification_target(df: pd.DataFrame,
                              price_col: str = "Close") -> pd.Series:
    """Binary target: 1 if next-day price is higher, else 0."""
    target = (df[price_col].shift(-1) > df[price_col]).astype(int)
    target.name = "Target"
    return target


def add_forecast_target(df: pd.DataFrame,
                        price_col: str = "Close",
                        horizon: int = 1) -> pd.Series:
    """Regression target: future price at *horizon* days."""
    target = df[price_col].shift(-horizon)
    target.name = f"FuturePrice_{horizon}d"
    return target


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(X_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> tuple[np.ndarray,
                                                   np.ndarray,
                                                   StandardScaler]:
    """Fit StandardScaler on train, transform both train and test."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler
