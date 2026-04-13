# O4 — Stock Predictor

An end-to-end Python machine learning pipeline for **stock price prediction** and **trading strategy simulation**.

## Overview

This project downloads historical stock data, engineers technical features, trains multiple ML/DL models to predict next-day price direction, evaluates them, runs a Prophet forecast, and back-tests a simple trading strategy.

### Stocks covered

| Ticker | Company |
|--------|---------|
| AAPL   | Apple |
| JPM    | JPMorgan Chase |
| TSLA   | Tesla |

Market indices **^VIX** and **^GSPC** (S&P 500) are also downloaded as supplemental features.

## Project structure

```
├── data.py          # Data loading (yfinance)
├── features.py      # Feature engineering (RSI, MACD, Bollinger, etc.)
├── models.py        # Model definitions (sklearn, Keras, Prophet)
├── train.py         # Training pipeline (time-series split, GridSearchCV)
├── evaluate.py      # Evaluation metrics (classification & forecasting)
├── strategy.py      # Trading simulation & back-testing
├── main.py          # Entry point — runs the full pipeline
├── tests/           # Unit tests (pytest)
├── requirements.txt # Python dependencies
└── README.md
```

## Features engineered

- OHLCV prices and volume
- Moving averages (10, 20, 50 days)
- RSI (14-period)
- MACD (12/26/9)
- Bollinger Bands (20-period, 2σ)
- Daily & 5-day returns
- Momentum (1-day, 5-day price diffs)
- Volatility (rolling 10-day, 20-day std of returns)
- Lagged close prices (1, 2, 3, 5 days)
- VIX index

## Models

### Classification (next-day direction)

| Model | Library |
|-------|---------|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| SVM | scikit-learn |
| XGBoost | xgboost |
| MLP | TensorFlow/Keras |
| LSTM | TensorFlow/Keras |
| Voting Ensemble | scikit-learn (soft voting) |

### Forecasting

| Model | Library |
|-------|---------|
| Prophet | prophet |

## Evaluation

**Classification**: Accuracy, Precision, Recall, F1-score, ROC-AUC

**Forecasting**: RMSE, MAE

**Trading strategy**: Total Return, Annualised Volatility, Sharpe Ratio, Maximum Drawdown

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Run tests
python -m pytest tests/ -v
```

## Output

Plots are saved to `./output/`:

- Feature importance bar charts
- SHAP summary plots (if available)
- Prophet forecast plots
- Predictions vs actual price
- Trading equity curves

All model metrics are printed to the console.