"""Unit tests for features.py — feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from features import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    build_features,
    add_classification_target,
    add_forecast_target,
    scale_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices() -> pd.Series:
    """100 days of synthetic close prices."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)
    prices = 100 * np.cumprod(1 + returns)
    idx = pd.bdate_range("2023-01-01", periods=100)
    return pd.Series(prices, index=idx, name="Close")


@pytest.fixture
def sample_ohlcv(sample_prices) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    close = sample_prices.values
    np.random.seed(0)
    return pd.DataFrame({
        "Open": close * (1 + np.random.uniform(-0.01, 0.01, len(close))),
        "High": close * (1 + np.abs(np.random.normal(0, 0.01, len(close)))),
        "Low": close * (1 - np.abs(np.random.normal(0, 0.01, len(close)))),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, len(close)),
    }, index=sample_prices.index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_output_range(self, sample_prices):
        rsi = compute_rsi(sample_prices).dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_length(self, sample_prices):
        rsi = compute_rsi(sample_prices, period=14)
        # First 13 values should be NaN
        assert rsi.iloc[:13].isna().all()


class TestMACD:
    def test_columns(self, sample_prices):
        macd_df = compute_macd(sample_prices)
        assert set(macd_df.columns) == {"MACD", "MACD_Signal", "MACD_Hist"}

    def test_histogram_identity(self, sample_prices):
        macd_df = compute_macd(sample_prices)
        diff = macd_df["MACD"] - macd_df["MACD_Signal"]
        np.testing.assert_allclose(diff.values, macd_df["MACD_Hist"].values,
                                   atol=1e-10)


class TestBollinger:
    def test_columns(self, sample_prices):
        bb = compute_bollinger(sample_prices)
        assert set(bb.columns) == {"BB_Upper", "BB_Mid", "BB_Lower"}

    def test_band_ordering(self, sample_prices):
        bb = compute_bollinger(sample_prices).dropna()
        assert (bb["BB_Upper"] >= bb["BB_Mid"]).all()
        assert (bb["BB_Mid"] >= bb["BB_Lower"]).all()


class TestBuildFeatures:
    def test_no_nan(self, sample_ohlcv):
        feat = build_features(sample_ohlcv)
        assert not feat.isna().any().any()

    def test_expected_columns_present(self, sample_ohlcv):
        feat = build_features(sample_ohlcv)
        expected = {"RSI", "MACD", "MA_10", "MA_20", "MA_50",
                    "Return_1d", "Volatility_10d", "Close_Lag1"}
        assert expected.issubset(set(feat.columns))


class TestTargets:
    def test_classification_target_binary(self, sample_ohlcv):
        target = add_classification_target(sample_ohlcv)
        assert set(target.dropna().unique()).issubset({0, 1})

    def test_forecast_target_shift(self, sample_ohlcv):
        target = add_forecast_target(sample_ohlcv, horizon=5)
        # The last 5 values should be NaN
        assert target.iloc[-5:].isna().all()
        assert target.iloc[0] == sample_ohlcv["Close"].iloc[5]


class TestScaling:
    def test_mean_std(self, sample_ohlcv):
        feat = build_features(sample_ohlcv)
        mid = len(feat) // 2
        train = feat.iloc[:mid]
        test = feat.iloc[mid:]
        X_tr, X_te, scaler = scale_features(train, test)
        # Training data should be roughly zero-mean, unit-variance
        np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=0.1)
        np.testing.assert_allclose(X_tr.std(axis=0), 1, atol=0.3)
