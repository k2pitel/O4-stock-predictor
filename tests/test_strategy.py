"""Unit tests for strategy.py — trading simulation."""

import numpy as np
import pandas as pd
import pytest

from strategy import simulate_strategy, strategy_metrics


@pytest.fixture
def simple_sim():
    """Simulate a trivial case: always long on steadily rising prices."""
    prices = pd.Series(np.linspace(100, 110, 20))
    signals = np.ones(20, dtype=int)
    return simulate_strategy(prices, signals)


class TestSimulateStrategy:
    def test_output_columns(self, simple_sim):
        expected = {"Price", "Signal", "Return", "StrategyReturn",
                    "Equity", "BuyHoldEquity"}
        assert expected == set(simple_sim.columns)

    def test_equity_increases_on_rising_prices(self, simple_sim):
        assert simple_sim["Equity"].iloc[-1] > 1.0

    def test_zero_signal_no_exposure(self):
        prices = pd.Series(np.linspace(100, 110, 20))
        signals = np.zeros(20, dtype=int)
        sim = simulate_strategy(prices, signals)
        # With no exposure, strategy return should be 0 each day
        np.testing.assert_allclose(sim["StrategyReturn"].values, 0.0)

    def test_length(self):
        prices = pd.Series(np.arange(50, dtype=float) + 100)
        signals = np.ones(50, dtype=int)
        sim = simulate_strategy(prices, signals)
        assert len(sim) == 49  # n-1 returns


class TestStrategyMetrics:
    def test_positive_return(self, simple_sim):
        m = strategy_metrics(simple_sim)
        assert m["Total Return"] > 0

    def test_max_drawdown_nonpositive(self, simple_sim):
        m = strategy_metrics(simple_sim)
        assert m["Max Drawdown"] <= 0

    def test_sharpe_finite(self, simple_sim):
        m = strategy_metrics(simple_sim)
        assert np.isfinite(m["Sharpe Ratio"])
