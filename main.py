"""
main.py — Entry point for the stock prediction pipeline.

Orchestrates:
  1. Data download
  2. Feature engineering
  3. Model training (classification, MLP, LSTM, ensemble)
  4. Evaluation & comparison
  5. Prophet forecasting (180-day horizon beyond current data)
  6. Trading strategy back-test
  7. Feature importance / SHAP
  8. Visualisation
"""

from __future__ import annotations

import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from data import load_data, TICKERS
from features import build_features, add_classification_target, add_forecast_target
from models import build_prophet, fit_prophet, predict_prophet, create_sequences
from train import run_training_pipeline
from evaluate import (
    evaluate_all_classifiers,
    evaluate_classifier,
    evaluate_keras_model,
    evaluate_lstm,
    forecast_metrics,
)
from strategy import simulate_strategy, strategy_metrics, plot_equity_curve


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
TUNE = True
LSTM_EPOCHS = 30
MLP_EPOCHS = 50
PROPHET_HORIZON = 180   # days to forecast into the future beyond current data
PROPHET_VALIDATION = 60 # days held out for back-test validation


# ---------------------------------------------------------------------------
# Design tokens — shared across all plots
# ---------------------------------------------------------------------------
PAL = {
    "blue":    "#3d7eff",
    "orange":  "#f59e0b",
    "green":   "#10b981",
    "red":     "#ef4444",
    "purple":  "#8b5cf6",
    "navy":    "#0f172a",
    "grid":    "#e2e8f0",
    "bg":      "#f8fafc",
    "text":    "#0f172a",
    "sub":     "#64748b",
    "border":  "#cbd5e1",
}


def setup_global_style() -> None:
    """Apply a consistent professional style to all matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor":     "white",
        "axes.facecolor":       PAL["bg"],
        "axes.grid":            True,
        "grid.alpha":           0.6,
        "grid.color":           PAL["grid"],
        "grid.linestyle":       "--",
        "grid.linewidth":       0.8,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.spines.left":     True,
        "axes.spines.bottom":   True,
        "axes.edgecolor":       PAL["border"],
        "axes.labelcolor":      PAL["sub"],
        "axes.titlecolor":      PAL["text"],
        "axes.titleweight":     "bold",
        "axes.titlesize":       13,
        "axes.labelsize":       9,
        "xtick.color":          PAL["sub"],
        "ytick.color":          PAL["sub"],
        "xtick.labelsize":      8,
        "ytick.labelsize":      8,
        "legend.frameon":       True,
        "legend.framealpha":    0.95,
        "legend.facecolor":     "white",
        "legend.edgecolor":     PAL["grid"],
        "legend.fontsize":      9,
        "font.family":          ["DejaVu Sans"],
        "figure.dpi":           150,
        "savefig.bbox":         "tight",
        "savefig.dpi":          150,
    })


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curve(history, model_name: str, ticker: str) -> None:
    """Plot training and validation loss + accuracy over epochs."""
    fig, (ax_loss, ax_acc) = plt.subplots(
        1, 2, figsize=(9, 6), facecolor="white"
    )
    epochs = range(1, len(history.history["loss"]) + 1)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, history.history["loss"],
                 color=PAL["blue"], linewidth=2.2, label="Train loss")
    if "val_loss" in history.history:
        ax_loss.plot(epochs, history.history["val_loss"],
                     color=PAL["orange"], linewidth=2.0,
                     linestyle="--", label="Validation loss")
    ax_loss.fill_between(epochs, history.history["loss"],
                         alpha=0.08, color=PAL["blue"])
    ax_loss.set_title(f"{model_name} — Loss  ({ticker})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary Cross-Entropy")
    ax_loss.legend()
    ax_loss.set_xlim(1, max(epochs))

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax_acc.plot(epochs, history.history["accuracy"],
                color=PAL["blue"], linewidth=2.2, label="Train accuracy")
    if "val_accuracy" in history.history:
        ax_acc.plot(epochs, history.history["val_accuracy"],
                    color=PAL["orange"], linewidth=2.0,
                    linestyle="--", label="Validation accuracy")
    ax_acc.axhline(0.5, color=PAL["red"], linewidth=1.2, linestyle=":",
                   alpha=0.7, label="Random baseline")
    ax_acc.fill_between(epochs, history.history["accuracy"],
                        alpha=0.08, color=PAL["blue"])
    ax_acc.set_title(f"{model_name} — Accuracy  ({ticker})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.set_xlim(1, max(epochs))
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"learning_curve_{model_name}_{ticker}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Learning curve saved to {path}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def show_feature_importance(trained: dict, feature_names: list[str]) -> None:
    """Plot feature importance for tree-based classifiers with gradient bars."""
    for name in ("RandomForest", "XGBoost"):
        clf = trained.get(name)
        if clf is None:
            continue
        importances = clf.feature_importances_
        top_n = 15
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_idx = indices[::-1]   # lowest → highest for horizontal bar

        vals = importances[sorted_idx]
        labels = [feature_names[i] for i in sorted_idx]

        # Color gradient: light → dark blue by importance
        norm_vals = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
        cmap = LinearSegmentedColormap.from_list(
            "imp", ["#bfdbfe", "#1d4ed8"], N=256
        )
        colors = [cmap(v) for v in norm_vals]

        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
        bars = ax.barh(labels, vals, color=colors, height=0.7, edgecolor="white")

        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=7.5, color=PAL["sub"],
            )

        ax.set_title(f"Feature Importance — {name}", fontsize=13, fontweight="bold",
                     color=PAL["text"])
        ax.set_xlabel("Importance Score", fontsize=9)
        ax.set_xlim(0, vals.max() * 1.18)
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()

        path = os.path.join(OUTPUT_DIR, f"feature_importance_{name}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Feature importance plot saved to {path}")


def try_shap(trained: dict, X_sample: np.ndarray,
             feature_names: list[str]) -> None:
    try:
        import shap
    except ImportError:
        print("  SHAP not available — skipping.")
        return

    for name in ("RandomForest", "XGBoost"):
        clf = trained.get(name)
        if clf is None:
            continue
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_sample[:200])
            fig, ax = plt.subplots(facecolor="white")
            shap.summary_plot(shap_values, X_sample[:200],
                              feature_names=feature_names, show=False)
            path = os.path.join(OUTPUT_DIR, f"shap_{name}.png")
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            print(f"  SHAP plot saved to {path}")
        except Exception as exc:
            print(f"  SHAP failed for {name}: {exc}")


# ---------------------------------------------------------------------------
# Prophet forecasting — validation + future forecast
# ---------------------------------------------------------------------------

def run_prophet(df: pd.DataFrame, ticker: str) -> None:
    """
    Two-phase Prophet workflow:
      1. Validation: train on all-but-last PROPHET_VALIDATION days, score on held-out.
      2. Future:     train on ALL data, forecast PROPHET_HORIZON days beyond last date.
    """
    close = df["Close"].copy()

    # ── Phase 1: validation metrics ──────────────────────────────────────────
    train_close = close.iloc[:-PROPHET_VALIDATION]
    test_close  = close.iloc[-PROPHET_VALIDATION:]

    val_model = build_prophet()
    val_model = fit_prophet(val_model, train_close.index, train_close.values)
    val_fc = predict_prophet(val_model, periods=PROPHET_VALIDATION)
    val_fc_idx = val_fc.set_index("ds").reindex(test_close.index)
    y_pred = val_fc_idx["yhat"].dropna().values
    y_true = test_close.iloc[: len(y_pred)].values
    if len(y_true) > 0:
        m = forecast_metrics(y_true, y_pred)
        print(f"  Prophet ({ticker}): RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")

    # ── Phase 2: full model + future forecast ────────────────────────────────
    full_model = build_prophet()
    full_model = fit_prophet(full_model, close.index, close.values)
    future_df  = predict_prophet(full_model, periods=PROPHET_HORIZON)
    fc = future_df.set_index("ds")

    last_date = close.index[-1]
    hist_fc   = fc[fc.index <= last_date]
    futr_fc   = fc[fc.index >  last_date]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor="white")

    # Historical actual
    ax.plot(close.index, close.values,
            color=PAL["blue"], linewidth=1.6, label="Actual Close", zorder=4)

    # In-sample fit (dashed, subtle)
    ax.plot(hist_fc.index, hist_fc["yhat"].values,
            color=PAL["orange"], linewidth=1.0, linestyle="--",
            alpha=0.55, label="Prophet in-sample fit", zorder=3)

    # Future forecast line + CI band
    ax.plot(futr_fc.index, futr_fc["yhat"].values,
            color=PAL["orange"], linewidth=2.5,
            label=f"Prophet forecast (+{PROPHET_HORIZON}d)", zorder=5)
    ax.fill_between(
        futr_fc.index,
        futr_fc["yhat_lower"].values, futr_fc["yhat_upper"].values,
        alpha=0.20, color=PAL["orange"], label="80% Confidence Interval",
    )

    # Vertical separator
    ax.axvline(last_date, color=PAL["sub"], linewidth=1.5,
               linestyle=":", alpha=0.8, zorder=6)
    ymin, ymax = ax.get_ylim()
    ax.text(last_date, ymax * 0.97, "  Forecast →",
            fontsize=8.5, color=PAL["sub"], va="top", fontstyle="italic")

    # Annotate forecast end-point
    if len(futr_fc) > 0:
        end_val = futr_fc["yhat"].iloc[-1]
        end_date = futr_fc.index[-1]
        ax.annotate(
            f"${end_val:.0f}",
            xy=(end_date, end_val),
            xytext=(-40, 12), textcoords="offset points",
            fontsize=9, fontweight="bold", color=PAL["orange"],
            arrowprops=dict(arrowstyle="->", color=PAL["orange"], lw=1.0),
        )

    ax.set_title(
        f"Prophet {PROPHET_HORIZON}-Day Price Forecast — {ticker}",
        fontsize=14, fontweight="bold", color=PAL["text"], pad=10,
    )
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right", fontsize=8)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    leg = ax.legend(loc="upper left", fontsize=9)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"prophet_{ticker}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Forecast plot saved to {path}")


# ---------------------------------------------------------------------------
# Predictions vs Actuals
# ---------------------------------------------------------------------------

def plot_predictions_vs_actual(
    prices_test: pd.Series,
    signals: np.ndarray,
    ticker: str,
) -> None:
    """Price series with coloured up/down markers for model predictions."""
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")

    x = np.arange(len(prices_test))
    ax.plot(x, prices_test.values, color=PAL["sub"], linewidth=1.2,
            alpha=0.55, label="Actual Close", zorder=2)

    up   = signals == 1
    down = ~up
    ax.scatter(x[up],   prices_test.values[up],
               marker="^", c=PAL["green"], s=18,
               label="Predicted Up ↑", alpha=0.8, zorder=4, linewidths=0)
    ax.scatter(x[down], prices_test.values[down],
               marker="v", c=PAL["red"], s=18,
               label="Predicted Down ↓", alpha=0.8, zorder=4, linewidths=0)

    # Directional accuracy annotation
    n_correct_up = np.sum(up)
    pct = n_correct_up / len(signals) * 100
    ax.text(0.98, 0.96, f"↑ signals: {pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=PAL["sub"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=PAL["grid"], alpha=0.9))

    ax.set_title(f"Predictions vs Actual Close — {ticker}",
                 fontsize=13, fontweight="bold", color=PAL["text"])
    ax.set_xlabel("Test-set Index", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    leg = ax.legend(fontsize=9, loc="upper left")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"pred_vs_actual_{ticker}.png")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_global_style()
    ensure_output_dir()

    print("\n" + "=" * 60)
    print("STEP 1: Downloading data")
    print("=" * 60)
    data = load_data()

    all_results: dict[str, pd.DataFrame] = {}

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {ticker}")
        print("=" * 60)

        df  = data[ticker]
        vix = data.get("^VIX")

        print("\nSTEP 2: Feature engineering")
        features = build_features(df, vix=vix)
        feature_names = list(features.columns)
        print(f"  Features shape: {features.shape}")

        print("\nSTEP 3: Training models")
        result = run_training_pipeline(
            features, tune=TUNE,
            lstm_epochs=LSTM_EPOCHS, mlp_epochs=MLP_EPOCHS,
        )

        print("\nSTEP 4: Evaluation")
        print("-" * 50)
        clf_results = evaluate_all_classifiers(
            result["classifiers"], result["X_test"], result["y_test"],
        )

        ens_metrics = evaluate_classifier(
            result["ensemble"], result["X_test"], result["y_test"],
            name="VotingEnsemble",
        )
        clf_results.loc["VotingEnsemble"] = ens_metrics

        mlp_metrics = evaluate_keras_model(
            result["mlp"], result["X_test"], result["y_test"], name="MLP",
        )
        clf_results.loc["MLP"] = mlp_metrics
        plot_learning_curve(result["mlp_history"], "MLP", ticker)

        lstm_metrics = evaluate_lstm(
            result["lstm"], result["X_test"], result["y_test"],
            seq_len=result["lstm_seq_len"],
        )
        if lstm_metrics:
            clf_results.loc["LSTM"] = lstm_metrics
        plot_learning_curve(result["lstm_history"], "LSTM", ticker)

        print(f"\n  === Classification summary ({ticker}) ===")
        print(clf_results.to_string())
        all_results[ticker] = clf_results

        print("\nSTEP 5: Feature importance")
        show_feature_importance(result["classifiers"], feature_names)
        try_shap(result["classifiers"], result["X_test"], feature_names)

        print("\nSTEP 6: Prophet forecasting")
        run_prophet(df, ticker)

        print("\nSTEP 7: Trading strategy")
        f1_col = clf_results["F1"].dropna()
        best_clf_name = f1_col.idxmax() if len(f1_col) > 0 else "XGBoost"
        best_clf = (result["classifiers"].get(best_clf_name)
                    or result.get("ensemble")
                    or list(result["classifiers"].values())[0])
        print(f"  Using {best_clf_name} for strategy signals")

        if hasattr(best_clf, "predict"):
            signals = best_clf.predict(result["X_test"])
        else:
            signals = (best_clf.predict(result["X_test"], verbose=0).ravel()
                       >= 0.5).astype(int)

        prices_test = result["X_test_raw"]["Close"]
        n = min(len(prices_test), len(signals))
        prices_test = prices_test.iloc[:n]
        signals     = signals[:n]

        sim = simulate_strategy(prices_test, signals)
        sm  = strategy_metrics(sim)
        print(f"  Total Return:  {sm['Total Return']:.4f}")
        print(f"  Volatility:    {sm['Annualised Volatility']:.4f}")
        print(f"  Sharpe Ratio:  {sm['Sharpe Ratio']:.4f}")
        print(f"  Max Drawdown:  {sm['Max Drawdown']:.4f}")

        equity_path = os.path.join(OUTPUT_DIR, f"equity_{ticker}.png")
        plot_equity_curve(sim, title=f"Equity Curve — {ticker}  |  ML Strategy vs Buy & Hold",
                          save_path=equity_path)
        plot_predictions_vs_actual(prices_test, signals, ticker)

    print("\n" + "=" * 60)
    print("FINAL COMPARISON ACROSS ALL STOCKS")
    print("=" * 60)
    for ticker, res in all_results.items():
        print(f"\n--- {ticker} ---")
        print(res.to_string())

    print("\n✓ Pipeline complete.  Plots saved to ./output/")


if __name__ == "__main__":
    main()
