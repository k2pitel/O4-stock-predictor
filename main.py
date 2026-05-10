"""
main.py — Entry point for the stock prediction pipeline.

Orchestrates:
  1. Data download
  2. Feature engineering
  3. Model training (classification, MLP, LSTM, ensemble)
  4. Evaluation & comparison
  5. Prophet forecasting
  6. Trading strategy back-test
  7. Feature importance / SHAP
  8. Visualisation
"""

from __future__ import annotations

import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF info/warning

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

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
PROPHET_HORIZON = 30  # days


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Learning curves (cost function / overfitting diagnostic)
# ---------------------------------------------------------------------------

def plot_learning_curve(history, model_name: str, ticker: str) -> None:
    """Plot training and validation loss + accuracy over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.history["loss"], label="Train loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val loss")
    axes[0].set_title(f"{model_name} — Loss ({ticker})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Cross-Entropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history.history["accuracy"], label="Train accuracy")
    if "val_accuracy" in history.history:
        axes[1].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[1].set_title(f"{model_name} — Accuracy ({ticker})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"learning_curve_{model_name}_{ticker}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Learning curve saved to {path}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def show_feature_importance(trained: dict, feature_names: list[str]) -> None:
    """Print and plot feature importance for tree-based classifiers."""
    for name in ("RandomForest", "XGBoost"):
        clf = trained.get(name)
        if clf is None:
            continue
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]

        print(f"\n  Top features ({name}):")
        for i, idx in enumerate(indices):
            print(f"    {i + 1:2d}. {feature_names[idx]:25s}  "
                  f"{importances[idx]:.4f}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(
            [feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]],
        )
        ax.set_title(f"Feature Importance — {name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, f"feature_importance_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved to {path}")


def try_shap(trained: dict, X_sample: np.ndarray,
             feature_names: list[str]) -> None:
    """Attempt SHAP analysis for tree models (non-critical)."""
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
            fig = plt.figure()
            shap.summary_plot(shap_values, X_sample[:200],
                              feature_names=feature_names, show=False)
            path = os.path.join(OUTPUT_DIR, f"shap_{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  SHAP plot saved to {path}")
        except Exception as exc:
            print(f"  SHAP failed for {name}: {exc}")


# ---------------------------------------------------------------------------
# Prophet forecasting
# ---------------------------------------------------------------------------

def run_prophet(df: pd.DataFrame, ticker: str) -> None:
    """Fit Prophet and evaluate forecast on the last PROPHET_HORIZON days."""
    close = df["Close"].copy()
    train_close = close.iloc[:-PROPHET_HORIZON]
    test_close = close.iloc[-PROPHET_HORIZON:]

    model = build_prophet()
    model = fit_prophet(model, train_close.index, train_close.values)
    forecast = predict_prophet(model, periods=PROPHET_HORIZON)

    # Align forecast with test dates
    forecast_tail = forecast.set_index("ds").reindex(test_close.index)
    y_pred = forecast_tail["yhat"].dropna().values
    y_true = test_close.iloc[: len(y_pred)].values

    if len(y_true) > 0:
        m = forecast_metrics(y_true, y_pred)
        print(f"  Prophet ({ticker}): RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(close.index, close.values, label="Actual", alpha=0.7)
    fc = forecast.set_index("ds")
    ax.plot(fc.index, fc["yhat"].values, label="Prophet Forecast", alpha=0.8)
    ax.fill_between(
        fc.index, fc["yhat_lower"].values, fc["yhat_upper"].values,
        alpha=0.15, label="Confidence Interval",
    )
    ax.set_title(f"Prophet Forecast — {ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"prophet_{ticker}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Forecast plot saved to {path}")


# ---------------------------------------------------------------------------
# Predictions vs Actuals plot
# ---------------------------------------------------------------------------

def plot_predictions_vs_actual(
    prices_test: pd.Series,
    signals: np.ndarray,
    ticker: str,
) -> None:
    """Scatter-style plot: mark predicted up/down on actual price series."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(prices_test))
    ax.plot(x, prices_test.values, label="Actual Close", alpha=0.7)
    up = signals == 1
    ax.scatter(x[up], prices_test.values[up], marker="^", c="green",
               s=12, label="Predicted Up", alpha=0.6)
    down = ~up
    ax.scatter(x[down], prices_test.values[down], marker="v", c="red",
               s=12, label="Predicted Down/Flat", alpha=0.6)
    ax.set_title(f"Predictions vs Actual — {ticker}")
    ax.set_xlabel("Test-set index")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"pred_vs_actual_{ticker}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_output_dir()

    # 1. Data
    print("\n" + "=" * 60)
    print("STEP 1: Downloading data")
    print("=" * 60)
    data = load_data()

    all_results: dict[str, pd.DataFrame] = {}

    for ticker in TICKERS:
        print(f"\n{'=' * 60}")
        print(f"PROCESSING: {ticker}")
        print("=" * 60)

        df = data[ticker]
        vix = data.get("^VIX")

        # 2. Features
        print("\nSTEP 2: Feature engineering")
        features = build_features(df, vix=vix)
        feature_names = list(features.columns)
        print(f"  Features shape: {features.shape}")

        # 3. Train
        print("\nSTEP 3: Training models")
        result = run_training_pipeline(
            features, tune=TUNE,
            lstm_epochs=LSTM_EPOCHS, mlp_epochs=MLP_EPOCHS,
        )

        # 4. Evaluate classifiers
        print("\nSTEP 4: Evaluation")
        print("-" * 50)
        clf_results = evaluate_all_classifiers(
            result["classifiers"], result["X_test"], result["y_test"],
        )

        # Ensemble
        ens_metrics = evaluate_classifier(
            result["ensemble"], result["X_test"], result["y_test"],
            name="VotingEnsemble",
        )
        clf_results.loc["VotingEnsemble"] = ens_metrics

        # MLP
        mlp_metrics = evaluate_keras_model(
            result["mlp"], result["X_test"], result["y_test"], name="MLP",
        )
        clf_results.loc["MLP"] = mlp_metrics
        plot_learning_curve(result["mlp_history"], "MLP", ticker)

        # LSTM
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

        # 5. Feature importance
        print("\nSTEP 5: Feature importance")
        show_feature_importance(result["classifiers"], feature_names)
        try_shap(result["classifiers"], result["X_test"], feature_names)

        # 6. Prophet
        print("\nSTEP 6: Prophet forecasting")
        run_prophet(df, ticker)

        # 7. Trading strategy (use best individual: XGBoost by default)
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
        signals = signals[:n]

        sim = simulate_strategy(prices_test, signals)
        sm = strategy_metrics(sim)
        print(f"  Total Return:  {sm['Total Return']:.4f}")
        print(f"  Volatility:    {sm['Annualised Volatility']:.4f}")
        print(f"  Sharpe Ratio:  {sm['Sharpe Ratio']:.4f}")
        print(f"  Max Drawdown:  {sm['Max Drawdown']:.4f}")

        equity_path = os.path.join(OUTPUT_DIR, f"equity_{ticker}.png")
        plot_equity_curve(sim, title=f"Equity Curve — {ticker}",
                          save_path=equity_path)

        plot_predictions_vs_actual(prices_test, signals, ticker)

    # 8. Cross-stock comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON ACROSS ALL STOCKS")
    print("=" * 60)
    for ticker, res in all_results.items():
        print(f"\n--- {ticker} ---")
        print(res.to_string())

    print("\n✓ Pipeline complete.  Plots saved to ./output/")


if __name__ == "__main__":
    main()
