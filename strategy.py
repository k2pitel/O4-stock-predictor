"""
strategy.py — Trading simulation module.

Implements a simple signal-based strategy:
  • Buy (go long) when the model predicts the next-day price will rise.
  • Stay flat otherwise.

Evaluates: total return, volatility, Sharpe ratio, maximum drawdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

def simulate_strategy(
    prices: pd.Series,
    signals: np.ndarray,
) -> pd.DataFrame:
    """Run a simple long-only strategy driven by *signals*.

    Parameters
    ----------
    prices : pd.Series
        Actual close prices aligned with ``signals``.
    signals : array-like of {0, 1}
        1 = buy / hold, 0 = stay out (cash).

    Returns
    -------
    pd.DataFrame with columns:
        Price, Signal, Return, StrategyReturn, Equity, BuyHoldEquity
    """
    signals = np.asarray(signals).ravel()
    prices = prices.values if isinstance(prices, pd.Series) else np.asarray(prices)

    n = min(len(prices), len(signals))
    prices = prices[:n]
    signals = signals[:n]

    daily_return = np.diff(prices) / prices[:-1]
    strategy_return = daily_return * signals[:-1]  # signal decides exposure

    equity = np.cumprod(1 + strategy_return)
    buy_hold = np.cumprod(1 + daily_return)

    result = pd.DataFrame({
        "Price": prices[1:],
        "Signal": signals[:-1],
        "Return": daily_return,
        "StrategyReturn": strategy_return,
        "Equity": equity,
        "BuyHoldEquity": buy_hold,
    })
    return result


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def strategy_metrics(sim: pd.DataFrame,
                     risk_free_rate: float = 0.0) -> dict:
    """Compute key strategy performance metrics.

    Parameters
    ----------
    sim : pd.DataFrame  (output of ``simulate_strategy``)
    risk_free_rate : float
        Annualised risk-free rate (default 0).

    Returns
    -------
    dict with Total Return, Volatility, Sharpe Ratio, Max Drawdown.
    """
    strat_ret = sim["StrategyReturn"]
    equity = sim["Equity"]

    if len(equity) == 0 or equity.iloc[0] == 0:
        total_return = 0.0
    else:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    ann_return = strat_ret.mean() * 252
    sharpe = ((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

    # Maximum drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "Total Return": total_return,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_equity_curve(sim: pd.DataFrame, title: str = "Strategy Equity Curve",
                     save_path: str | None = None) -> None:
    """Plot strategy equity vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sim["Equity"].values, label="Strategy")
    ax.plot(sim["BuyHoldEquity"].values, label="Buy & Hold", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Equity (normalised)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Equity curve saved to {save_path}")
    plt.close(fig)
