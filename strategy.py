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
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Strategy simulation
# ---------------------------------------------------------------------------

def simulate_strategy(
    prices: pd.Series,
    signals: np.ndarray,
) -> pd.DataFrame:
    """Run a simple long-only strategy driven by *signals*."""
    signals = np.asarray(signals).ravel()
    prices = prices.values if isinstance(prices, pd.Series) else np.asarray(prices)

    n = min(len(prices), len(signals))
    prices = prices[:n]
    signals = signals[:n]

    daily_return = np.diff(prices) / prices[:-1]
    strategy_return = daily_return * signals[:-1]

    equity = np.cumprod(1 + strategy_return)
    buy_hold = np.cumprod(1 + daily_return)

    return pd.DataFrame({
        "Price": prices[1:],
        "Signal": signals[:-1],
        "Return": daily_return,
        "StrategyReturn": strategy_return,
        "Equity": equity,
        "BuyHoldEquity": buy_hold,
    })


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def strategy_metrics(sim: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    strat_ret = sim["StrategyReturn"]
    equity = sim["Equity"]

    if len(equity) == 0 or equity.iloc[0] == 0:
        total_return = 0.0
    else:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    ann_return = strat_ret.mean() * 252
    sharpe = ((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else 0.0

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
# Plotting — equity curve with drawdown panel
# ---------------------------------------------------------------------------

_C = {
    "blue": "#3d7eff",
    "orange": "#f59e0b",
    "red": "#ef4444",
    "grid": "#e2e8f0",
    "bg": "#f8fafc",
    "text": "#0f172a",
    "sub": "#64748b",
}


def _apply_base_style(ax) -> None:
    ax.set_facecolor(_C["bg"])
    ax.grid(True, color=_C["grid"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(colors=_C["sub"], labelsize=8)
    ax.yaxis.label.set_color(_C["sub"])
    ax.xaxis.label.set_color(_C["sub"])


def plot_equity_curve(sim: pd.DataFrame, title: str = "Strategy Equity Curve",
                      save_path: str | None = None) -> None:
    """Two-panel chart: equity curve (top) + drawdown (bottom)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True,
        facecolor="white",
    )

    x = np.arange(len(sim))
    eq = sim["Equity"].values
    bh = sim["BuyHoldEquity"].values

    # ── Upper panel: equity ──────────────────────────────────────────────────
    _apply_base_style(ax1)
    ax1.plot(x, eq, color=_C["blue"], linewidth=2.2, label="ML Strategy", zorder=3)
    ax1.plot(x, bh, color=_C["orange"], linewidth=2.0, alpha=0.85,
             label="Buy & Hold", linestyle="--", zorder=2)
    ax1.fill_between(x, 1, eq, where=(eq >= 1),
                     alpha=0.10, color=_C["blue"], zorder=1)
    ax1.fill_between(x, 1, eq, where=(eq < 1),
                     alpha=0.10, color=_C["red"], zorder=1)
    ax1.axhline(1.0, color="#94a3b8", linewidth=1.0, linestyle=":", alpha=0.8)

    # Annotate final returns
    final_strat = eq[-1]
    final_bh = bh[-1]
    pct_s = (final_strat - 1) * 100
    pct_b = (final_bh - 1) * 100
    sign_s = "+" if pct_s >= 0 else ""
    sign_b = "+" if pct_b >= 0 else ""
    ax1.annotate(
        f"{sign_s}{pct_s:.1f}%",
        xy=(x[-1], final_strat), xytext=(8, 0), textcoords="offset points",
        fontsize=10, fontweight="bold", color=_C["blue"], va="center",
    )
    ax1.annotate(
        f"{sign_b}{pct_b:.1f}%",
        xy=(x[-1], final_bh), xytext=(8, 0), textcoords="offset points",
        fontsize=10, fontweight="bold", color=_C["orange"], va="center",
    )

    ax1.set_title(title, fontsize=14, fontweight="bold", color=_C["text"], pad=10)
    ax1.set_ylabel("Equity (normalised to 1.0)", fontsize=9)
    leg = ax1.legend(fontsize=10, loc="upper left",
                     framealpha=0.95, edgecolor="#e2e8f0")
    leg.get_frame().set_linewidth(0.8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))

    # ── Lower panel: drawdown ────────────────────────────────────────────────
    _apply_base_style(ax2)
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max * 100

    ax2.fill_between(x, 0, drawdown, alpha=0.55, color=_C["red"])
    ax2.plot(x, drawdown, color=_C["red"], linewidth=1.0)
    ax2.axhline(0, color="#94a3b8", linewidth=0.8)

    max_dd = drawdown.min()
    max_dd_idx = int(np.argmin(drawdown))
    ax2.annotate(
        f"{max_dd:.1f}%",
        xy=(max_dd_idx, max_dd), xytext=(6, -8), textcoords="offset points",
        fontsize=8, color=_C["red"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=_C["red"], lw=0.8),
    )

    ax2.set_ylabel("Drawdown %", fontsize=9)
    ax2.set_xlabel("Trading Days", fontsize=9)
    ax2.set_ylim(top=2)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Equity curve saved to {save_path}")
    plt.close(fig)
