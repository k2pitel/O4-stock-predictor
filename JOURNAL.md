# O4 — Stock Predictor: Project Journal

**Author:** Kevin  
**Date:** May 13, 2026  
**Course Assignment:** O4  
**Project:** End-to-End Machine Learning Pipeline for Stock Price Prediction and Trading Strategy Simulation

---

## Table of Contents

1. [Project Motivation and Goals](#1-project-motivation-and-goals)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Acquisition — `data.py`](#3-data-acquisition--datapy)
4. [Feature Engineering — `features.py`](#4-feature-engineering--featurespy)
5. [Model Definitions — `models.py`](#5-model-definitions--modelspy)
6. [Training Pipeline — `train.py`](#6-training-pipeline--trainpy)
7. [Evaluation Framework — `evaluate.py`](#7-evaluation-framework--evaluatepy)
8. [Trading Strategy Simulation — `strategy.py`](#8-trading-strategy-simulation--strategypy)
9. [Orchestration and Visualisation — `main.py`](#9-orchestration-and-visualisation--mainpy)
10. [Testing — `tests/`](#10-testing--tests)
11. [Design Decisions and Tradeoffs](#11-design-decisions-and-tradeoffs)
12. [Challenges Encountered](#12-challenges-encountered)
13. [Results and Observations](#13-results-and-observations)
14. [Reflections on the ML Models](#14-reflections-on-the-ml-models)
15. [Future Work](#15-future-work)
16. [Conclusion](#16-conclusion)

---

## 1. Project Motivation and Goals

Stock market prediction is one of the most studied and debated application areas for machine learning. The appeal is obvious: financial markets generate enormous amounts of structured, time-stamped numerical data, making them superficially well-suited to pattern-recognition algorithms. At the same time, markets are notoriously difficult to predict because they aggregate the beliefs and actions of millions of rational (and irrational) participants, incorporating information almost instantly. This tension makes the domain intellectually rich and technically challenging.

The goal of this project was not to build a profitable trading system — that would be a far more serious engineering and financial undertaking. Instead, the goal was to build a complete, production-style machine learning pipeline that covers every stage of the ML lifecycle applied to financial data:

- **Data acquisition**: pulling real historical price data from a publicly accessible source.
- **Feature engineering**: transforming raw price series into a meaningful set of predictive signals drawn from classical technical analysis.
- **Model training**: training multiple classes of models — linear classifiers, tree ensembles, gradient boosting, feedforward neural networks, recurrent neural networks, and a time-series forecasting model.
- **Evaluation**: measuring model performance with appropriate metrics, avoiding the common pitfall of data leakage through proper time-based splitting.
- **Strategy simulation**: translating classification signals into buy/sell decisions and evaluating the resulting portfolio performance.
- **Visualisation**: producing publication-quality charts that make results interpretable at a glance.

Three stocks were chosen to provide diversity across sectors and behavioural profiles:
- **AAPL** (Apple Inc.): a large-cap technology stock with relatively smooth trending behaviour.
- **JPM** (JPMorgan Chase): a financial sector stock, sensitive to interest-rate environments and macroeconomic cycles.
- **TSLA** (Tesla Inc.): a highly volatile growth stock with strong retail investor sentiment, presenting a harder prediction target.

Market indices **^VIX** (the CBOE Volatility Index, a fear gauge) and **^GSPC** (the S&P 500) were downloaded as supplemental contextual features.

---

## 2. High-Level Architecture

The project follows a strictly modular design. Each concern is isolated into its own Python module, and the `main.py` entry point acts purely as an orchestrator, calling into the modules in a defined sequence.

```
main.py
  │
  ├── data.py          ← download OHLCV data via yfinance
  ├── features.py      ← build technical indicator feature matrix
  ├── models.py        ← model definitions (sklearn, Keras, Prophet)
  ├── train.py         ← split, scale, tune, and train all models
  ├── evaluate.py      ← compute classification and regression metrics
  └── strategy.py      ← simulate trading strategy and compute risk metrics
```

This separation of concerns has several practical benefits. Individual modules can be unit-tested in isolation without loading TensorFlow or downloading data. The feature engineering logic is reusable independently of any specific model. Adding a new model requires only extending `models.py` and wiring it up in `train.py`, with no changes elsewhere.

The pipeline runs entirely offline after the initial data download, which matters for reproducibility: the downloaded data is fixed for a given date range, so model comparisons are always performed against the same observations.

---

## 3. Data Acquisition — `data.py`

### 3.1 Design

The data layer is intentionally thin. Its only responsibility is to download and lightly clean OHLCV (Open, High, Low, Close, Volume) data for a list of ticker symbols and return it as a dictionary mapping each symbol to a Pandas DataFrame with a `DatetimeIndex`.

```python
TICKERS = ["AAPL", "JPM", "TSLA"]
MARKET_TICKERS = ["^VIX", "^GSPC"]
DEFAULT_START = "2016-01-01"
DEFAULT_END   = "2026-05-10"
```

The date range spans approximately a decade, capturing multiple distinct market regimes: the bull market of 2016–2019, the COVID-19 crash of March 2020, the subsequent aggressive recovery, the 2022 rate-hike bear market, and the AI-driven tech rally of 2023–2025. This variety in regimes is important because a model that only trains on a single-regime bull market will have very different characteristics from one trained across varied conditions.

### 3.2 The `yfinance` Library

Yahoo Finance's unofficial API, accessed through the `yfinance` Python library, provides adjusted daily OHLCV data. The `auto_adjust=True` parameter is used so that prices are automatically adjusted for splits and dividends, ensuring that historical price series are comparable across time and that percentage-change calculations are meaningful.

A subtle but important detail: `yfinance` sometimes returns a `MultiIndex` column structure depending on how many tickers are downloaded simultaneously. The code defensively handles this by detecting `isinstance(df.columns, pd.MultiIndex)` and flattening the columns accordingly. Neglecting this produces silently broken DataFrames.

### 3.3 Forward-Fill Cleaning

After downloading, the `load_data` function applies `ffill()` (forward fill) followed by `dropna()`. Forward-filling addresses missing trading days (e.g., market holidays) that might appear when aligning multiple series to a common index. Remaining NaNs — which would exist only at the very start of the series before any data appears — are then dropped.

The alternative to forward-fill would be to use only dates where all tickers have data (inner join on the date index). Forward-fill was chosen because it preserves the natural calendar alignment and avoids silently losing significant portions of data.

---

## 4. Feature Engineering — `features.py`

Feature engineering is arguably the most consequential module in the pipeline. No matter how powerful a model is, it can only learn from the information provided to it. The features implemented here are all drawn from classical technical analysis — a body of market analysis methods based on the idea that price history contains informative patterns.

### 4.1 Moving Averages (MA)

Three simple moving averages are computed: 10-day, 20-day, and 50-day windows.

```python
for w in [10, 20, 50]:
    feat[f"MA_{w}"] = close.rolling(window=w).mean()
```

Moving averages smooth out short-term noise and reveal the underlying trend direction. The relationship between price and its moving average (is price above or below the MA?) is one of the oldest signals in technical analysis. The three window lengths provide short-term, medium-term, and longer-term trend context simultaneously.

### 4.2 RSI — Relative Strength Index

The RSI is a momentum oscillator that measures the speed and change of price movements on a scale of 0 to 100.

```python
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

A value above 70 traditionally indicates overbought conditions; below 30 indicates oversold. The implementation uses a 14-period window, which is the standard parameter. A guard against division by zero is present: when `avg_loss` is exactly zero (a run of purely up days), the RS ratio is replaced with NaN rather than infinity. This produces a NaN RSI value for that day, which is later dropped.

### 4.3 MACD — Moving Average Convergence Divergence

MACD captures the relationship between two exponential moving averages of different speeds.

```python
def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    ...
```

Three derived columns are produced: the MACD line itself (fast EMA minus slow EMA), the signal line (a 9-period EMA of the MACD line), and the histogram (their difference). The histogram is particularly informative as it shows the rate of change of the MACD, giving early warning of momentum shifts.

The parameter `adjust=False` in `ewm()` means the exponential weights are computed using the recursive formula, which is computationally efficient and matches the industry-standard definition.

### 4.4 Bollinger Bands

Bollinger Bands place upper and lower envelopes around a 20-day simple moving average, separated by two standard deviations.

```python
def compute_bollinger(series, period=20, num_std=2.0):
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return pd.DataFrame({
        "BB_Upper": mid + num_std * std,
        "BB_Mid": mid,
        "BB_Lower": mid - num_std * std,
    })
```

The bands contract during quiet periods and expand during volatile ones. Prices touching the upper band suggest overbought conditions; touching the lower band suggests oversold. By providing all three bands as features, the model can learn the distance of the current price from each boundary.

### 4.5 Returns and Momentum

Both 1-day and 5-day percentage returns are computed, as are 1-day and 5-day momentum values (raw price differences). Returns capture the relative magnitude of moves; momentum captures the absolute direction and size. Together they provide information about recent market velocity.

### 4.6 Volatility Features

Rolling standard deviation of 1-day returns over 10-day and 20-day windows serves as a measure of recent market turbulence. Volatile periods often cluster (GARCH effects), and including realised volatility as a feature allows the model to recognise that its predictions should be more uncertain when volatility is high.

### 4.7 Lagged Close Prices

Close prices lagged 1, 2, 3, and 5 days provide direct autoregressive features. Even if the model cannot predict direction from other features, the lagged prices give it raw material to detect autocorrelation patterns in the price series.

### 4.8 VIX

The VIX (CBOE Volatility Index) measures the implied volatility of S&P 500 options and represents the market's collective expectation of near-term risk. It is a cross-asset feature, applying the same market-wide fear signal to all three stocks. It is reindexed to align with each stock's date index and forward-filled.

### 4.9 NaN Management

Because different features have different warm-up periods (the 50-day moving average requires 50 observations before producing a valid value), the feature matrix has NaN rows at the beginning. These are dropped at the end of `build_features`. This means that approximately the first 50–60 rows of each stock's data are discarded, which is a reasonable trade-off given that the total dataset spans nearly a decade.

### 4.10 Classification and Forecast Targets

Two target functions are provided:

- `add_classification_target`: produces a binary label — 1 if tomorrow's close is higher than today's, 0 otherwise. This is the label used by all classifier models.
- `add_forecast_target`: produces the raw future price at a configurable horizon. This is not used by the classifiers but is available for regression-type models.

The classification target is defined using `shift(-1)`, which looks one step into the future. This naturally produces a NaN in the final row (there is no "tomorrow" for the last observation), which must be handled before training.

---

## 5. Model Definitions — `models.py`

The model layer defines all machine learning architectures but does not perform any training. This separation keeps model construction clean and testable.

### 5.1 Logistic Regression

```python
LogisticRegression(max_iter=1000, random_state=42)
```

Logistic regression serves as the linear baseline. It assumes a linear decision boundary in feature space, which is almost certainly incorrect for financial data, but it provides a lower bound on performance: any more complex model that fails to outperform logistic regression is likely overfitting or improperly regularised.

`max_iter=1000` prevents the solver from giving up before converging on the sometimes ill-conditioned financial feature matrices.

### 5.2 Random Forest

```python
RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
```

Random forests are ensembles of decision trees, each trained on a bootstrap sample of the data and a random subset of features. The averaging across 200 trees dramatically reduces variance compared to a single tree. `n_jobs=-1` parallelises tree construction across all available CPU cores.

Random forests are particularly valuable in this pipeline because they produce feature importance scores via the mean decrease in impurity across all trees, which are later visualised to understand which technical indicators are most informative.

### 5.3 Support Vector Machine

```python
SVC(probability=True, random_state=42)
```

The SVM finds the maximum-margin hyperplane separating the two classes. `probability=True` wraps the SVM in a Platt scaling step, producing calibrated probability estimates required for soft-voting in the ensemble and for ROC-AUC calculation. SVMs are sensitive to feature scaling, which is why the StandardScaler in `train.py` is essential.

### 5.4 XGBoost

```python
XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
              eval_metric="logloss", random_state=42)
```

XGBoost (Extreme Gradient Boosting) constructs trees sequentially, with each tree correcting the errors of the previous ensemble. It is the most powerful of the sklearn-compatible classifiers in the pipeline and often achieves the best classification metrics. Like Random Forest, it also exposes feature importances — but using the gain-based metric rather than impurity decrease, producing different importance rankings that reveal complementary information.

### 5.5 Voting Ensemble

```python
VotingClassifier(estimators=..., voting="soft", n_jobs=-1)
```

The soft-voting ensemble averages the class probability estimates from all four base classifiers (Logistic Regression, Random Forest, SVM, XGBoost). Averaging predictions reduces variance and tends to smooth out the individual errors of each constituent model. The soft vote (averaging probabilities) is typically superior to hard vote (majority label) because it weights confident predictions more heavily.

### 5.6 MLP — Multilayer Perceptron

```python
def build_mlp(input_dim, hidden_units=(128, 64), dropout=0.3):
    model = keras.Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
```

The MLP receives a flat feature vector and passes it through two hidden layers (128 and 64 units respectively) with ReLU activations and 30% dropout. Dropout randomly zeros out neurons during training, acting as a powerful regulariser that prevents co-adaptation of neurons. The output layer uses sigmoid to produce a scalar probability in [0, 1].

The architecture choice (two hidden layers) reflects a balance: deep enough to capture non-linearities in the feature space, but shallow enough to train quickly on the relatively small tabular financial dataset. A model with six hidden layers would almost certainly overfit.

### 5.7 LSTM — Long Short-Term Memory

```python
def build_lstm(seq_len, n_features, units=64, dropout=0.2):
    model = keras.Sequential()
    model.add(Input(shape=(seq_len, n_features)))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
```

The LSTM receives sequences of 30 consecutive trading days and learns temporal dependencies across that window. Unlike the MLP, which treats each day's features as independent, the LSTM can theoretically learn patterns like "when volatility has been rising for 5 consecutive days, the market is more likely to reverse." The `create_sequences` function in `models.py` prepares the 3D input tensor `(n_samples, seq_len, n_features)` required by the LSTM.

The LSTM layer with `return_sequences=False` only passes the final hidden state to the next layer, which is appropriate for sequence classification (as opposed to sequence-to-sequence mapping).

### 5.8 Prophet

```python
Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
```

Facebook's Prophet is a decomposable time-series model that fits an additive model with trend, seasonality (yearly and weekly), and holidays. It is fundamentally different from the classifiers: it is a univariate forecasting model that predicts future price levels directly rather than predicting the direction of next-day movement.

`daily_seasonality=False` is appropriate because the input data is already at daily granularity (one bar per trading day), so there is no intra-day pattern to model.

---

## 6. Training Pipeline — `train.py`

### 6.1 Time-Series Train/Test Split

Financial data has a critical constraint that sets it apart from most machine learning applications: observations are not exchangeable. The temporal ordering matters because a model trained on data from 2024 should not use features from 2025 as training examples — that would be data leakage.

```python
def time_series_train_test_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]
```

The split is strictly chronological: the most recent 20% of observations become the test set. This means the model is trained on older data and evaluated on more recent data, simulating realistic deployment conditions. The test set represents approximately two years of trading data (given the full 2016–2026 range).

This approach is in contrast to random train/test splits, which would be incorrect here: random splitting would allow features from "the future" (relative to some test points) to appear in the training set, leading to wildly optimistic accuracy estimates.

### 6.2 Feature Scaling

```python
X_train_sc, X_test_sc, scaler = scale_features(X_train_raw, X_test_raw)
```

StandardScaler is fit **only on the training data** and then applied (transform only) to the test data. This is essential: fitting the scaler on all data would allow information about the test set's distribution to influence the normalisation applied to training features, constituting a subtle form of data leakage.

Scaling is necessary for Logistic Regression, SVM, and MLP — these models are sensitive to feature magnitudes. XGBoost and Random Forest are invariant to monotonic transformations of features, so scaling has no effect on them, but it does no harm.

### 6.3 Hyperparameter Tuning with GridSearchCV

For three of the four sklearn classifiers, a compact grid search is performed using `TimeSeriesSplit` cross-validation:

```python
PARAM_GRIDS = {
    "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}
```

`TimeSeriesSplit` performs cross-validation respecting temporal ordering: each fold's training set consists of all data before the validation period, never including future data. This is the correct approach for time-series hyperparameter tuning; standard K-fold cross-validation would again risk leakage.

The SVM is excluded from grid search because its training complexity is O(n²) to O(n³) with kernel methods, making grid search prohibitively expensive on large datasets.

The `TUNE = True` flag in `main.py` controls whether tuning is performed. Setting it to `False` uses default parameters and significantly reduces runtime, which is useful during development iteration.

### 6.4 Deep Learning Training

MLP and LSTM are trained with 10% validation splits (drawn from the end of the training set, maintaining temporal order). Their training histories are captured and later used to plot learning curves — a diagnostic tool showing whether models are still improving or have converged (and whether they are overfitting to training data).

LSTM training requires the `create_sequences` function, which rolls a 30-day window over the flattened training set to produce overlapping input sequences. This means the effective number of training examples for the LSTM is `n_train - 30 + 1`, slightly smaller than the other models' training set.

---

## 7. Evaluation Framework — `evaluate.py`

### 7.1 Classification Metrics

Five standard metrics are computed for each model:

| Metric | Formula | Relevance |
|--------|---------|-----------|
| **Accuracy** | (TP + TN) / Total | Proportion of correct predictions |
| **Precision** | TP / (TP + FP) | Of all "Up" predictions, how many were correct |
| **Recall** | TP / (TP + FN) | Of all true "Up" days, how many were caught |
| **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve | Discrimination ability across all thresholds |

Accuracy alone is misleading if the class distribution is imbalanced (e.g., if the market rises 55% of days, a model that always predicts "Up" achieves 55% accuracy without learning anything). The F1-score and ROC-AUC give a more complete picture, and best F1 is used in `main.py` to select which model drives the trading strategy.

### 7.2 Evaluation Without Data Leakage

All metrics are computed on the held-out test set. For the LSTM, `create_sequences` is applied to the test features using the same sequence length as training, ensuring the evaluation matches training conditions.

### 7.3 Forecasting Metrics

For Prophet (a regression model predicting price levels), two metrics are used:

- **RMSE** (Root Mean Squared Error): penalises large errors heavily, making it sensitive to outliers.
- **MAE** (Mean Absolute Error): the average absolute deviation in the same units as the price (dollars), making it directly interpretable.

The validation procedure holds out the last 60 trading days, fits Prophet on all earlier data, forecasts 60 days forward, and computes metrics against the held-out actuals.

---

## 8. Trading Strategy Simulation — `strategy.py`

### 8.1 Strategy Logic

The strategy is a simple long-only signal-follower:

- When the best classifier predicts the next day's price will rise (signal = 1), take a position (be "in" the market).
- When the model predicts a down day (signal = 0), stay flat (out of the market).

```python
daily_return = np.diff(prices) / prices[:-1]
strategy_return = daily_return * signals[:-1]
equity = np.cumprod(1 + strategy_return)
buy_hold = np.cumprod(1 + daily_return)
```

The equity curve is normalised to start at 1.0, making it easy to compare strategies with different starting capitals. A benchmark buy-and-hold equity curve is computed alongside, which always holds the stock regardless of the model's signal.

This is a simplified model with several important caveats: it ignores transaction costs (brokerage fees, spreads), assumes fractional shares, assumes the end-of-day close price is the execution price, and ignores slippage. Real-world strategy evaluation would require accounting for all of these.

### 8.2 Performance Metrics

Four standard portfolio risk metrics are computed:

**Total Return**: the final equity value minus the initial (which is 1.0), expressed as a decimal. A total return of 0.35 means the strategy grew capital by 35%.

**Annualised Volatility**: the standard deviation of daily strategy returns scaled to annual by multiplying by √252 (the approximate number of trading days in a year). This quantifies how much the equity curve fluctuates.

**Sharpe Ratio**: the annualised excess return divided by annualised volatility. In this implementation the risk-free rate is set to 0 for simplicity (a common approximation). A Sharpe above 1.0 is generally considered good; above 2.0 is excellent; negative Sharpe means the strategy underperforms holding cash.

**Maximum Drawdown**: the largest peak-to-trough decline in the equity curve expressed as a percentage. It measures the worst loss an investor would have experienced if they bought at the highest point before the drawdown and sold at the bottom. Maximum drawdown is critical for assessing the psychological and practical risk of a strategy.

### 8.3 Visualisation

The equity curve plot uses a two-panel layout: the upper panel shows the ML strategy equity overlaid with buy-and-hold, with shaded regions distinguishing profitable periods (blue fill above 1.0) from loss periods (red fill below 1.0). The lower panel shows the running drawdown as a red filled area, annotated with the maximum drawdown value at its trough.

---

## 9. Orchestration and Visualisation — `main.py`

### 9.1 Pipeline Flow

`main.py` orchestrates the full pipeline in a single sequential pass through seven steps for each ticker:

1. **Download data** (once, before the per-ticker loop)
2. **Feature engineering**
3. **Model training** (sklearn classifiers, ensemble, MLP, LSTM)
4. **Evaluation** (all classifiers, ensemble, MLP, LSTM, summary table)
5. **Feature importance** (bar charts and SHAP plots)
6. **Prophet forecasting** (validation metrics + future forecast plot)
7. **Trading strategy** (simulate, compute metrics, plot equity curve and predictions vs actuals)

The loop runs three times — once for AAPL, once for JPM, once for TSLA — with all results stored in an `all_results` dictionary for a final cross-stock comparison table at the end.

### 9.2 Global Style System

A design token dictionary (`PAL`) establishes a consistent visual identity across all plots:

```python
PAL = {
    "blue":   "#3d7eff",
    "orange": "#f59e0b",
    "green":  "#10b981",
    "red":    "#ef4444",
    ...
}
```

These tokens are applied through `setup_global_style()`, which configures `matplotlib.rcParams` globally before any plot is drawn. The result is that all 18+ output plots share the same colour palette, font sizes, grid style, and background, giving the project a cohesive and professional appearance.

Using `matplotlib.use("Agg")` ensures that Matplotlib operates in non-interactive mode, writing files directly without attempting to open a display window. This is necessary for running the pipeline in server or headless environments.

### 9.3 Learning Curve Plots

For both MLP and LSTM, training history (loss and accuracy per epoch) is captured and plotted with dual panels: the left panel shows training vs. validation loss over epochs; the right shows training vs. validation accuracy with a dashed 50% random-baseline reference line.

These plots are diagnostic: if training loss continues to fall while validation loss rises, the model is overfitting. If both curves plateau early, the model has underfit and could benefit from more capacity or features.

### 9.4 Prophet Forecasting Plot

The Prophet plot shows:
- The full historical actual close price (blue line).
- Prophet's in-sample fit over historical data (dashed orange, subtle).
- The 180-day future forecast (solid orange, prominent).
- The 80% confidence interval around the forecast (orange shaded band).
- A vertical separator at the last date of observed data.
- A price annotation at the forecast endpoint.

The 80% confidence interval is critical for communicating uncertainty: a narrow band would be overconfident; a wide band correctly reflects that forecasting is inherently uncertain, especially over a 180-day horizon.

### 9.5 SHAP Analysis

If the `shap` library is available, SHAP (SHapley Additive exPlanations) values are computed for Random Forest and XGBoost. SHAP values provide a game-theoretically grounded decomposition of each prediction into per-feature contributions, making model behaviour interpretable at the individual prediction level. The summary plot shows the distribution of SHAP values for each feature across all test samples, revealing not only which features matter most but also whether high or low values of a feature push the prediction toward Up or Down.

---

## 10. Testing — `tests/`

The project includes four test modules with comprehensive unit coverage using `pytest`.

### 10.1 `test_features.py`

Tests cover all feature engineering functions:

- **RSI**: validates output range [0, 100] and that the first 13 values are NaN (the 14-period warm-up period).
- **MACD**: validates column names and the algebraic identity `MACD_Hist = MACD - MACD_Signal`.
- **Bollinger Bands**: validates column names and the ordering invariant `Upper >= Mid >= Lower`.
- **`build_features`**: validates that no NaN values remain after the function runs, and that expected columns are present.
- **Classification target**: validates binary output {0, 1}.
- **Forecast target**: validates the shift — target[0] should equal price[5] for a 5-day horizon.
- **Scaling**: validates that the scaled training set has approximately zero mean and unit standard deviation per feature.

The tests use synthetic data generated with a seeded random number generator, ensuring they are fast, deterministic, and independent of network access.

### 10.2 `test_strategy.py`

Tests cover the simulation and metrics functions:

- **Output columns**: validates the DataFrame schema of `simulate_strategy`.
- **Positive return on rising prices**: when prices rise monotonically and the signal is always long, equity must exceed 1.0.
- **Zero-signal no-exposure**: when all signals are 0 (always flat), all strategy returns must be exactly 0.
- **Length invariant**: for n price points, the simulation produces n-1 return rows (since returns require two consecutive prices).
- **Max drawdown non-positive**: drawdown by definition cannot be positive.
- **Sharpe finiteness**: metrics must not produce NaN or infinity.

### 10.3 `test_models.py` and `test_evaluate.py`

These modules test that model constructors return properly configured instances, that sequence creation produces the correct shapes, that classification metrics are within valid ranges, and that RMSE/MAE produce non-negative values.

The tests are deliberately lightweight: they validate contracts and invariants rather than performing end-to-end training, keeping the test suite fast enough to run in seconds.

---

## 11. Design Decisions and Tradeoffs

### 11.1 Classification vs. Regression

A natural question is why next-day direction (binary classification) was chosen as the primary task rather than next-day price level (regression). The reason is that price levels are non-stationary — they drift over long time horizons. A model trained to predict absolute price levels will fail when prices enter previously unseen ranges. Direction, by contrast, is a stationary quantity, and percentage returns are approximately stationary. The classification framing also directly translates to actionable trading signals.

### 11.2 Why Not Use All Available Data for Training?

An 80/20 split leaves 20% of the data strictly for testing. A common temptation is to train on all data and report "training accuracy," but this would be meaningless — any model with enough capacity will perfectly memorise training data. The held-out test set is the only honest measure of generalisation.

### 11.3 Soft vs. Hard Voting in the Ensemble

Soft voting (averaging probabilities) was chosen over hard voting (majority label) because it allows confident predictions to dominate. If three models give 90% probability of "Up" and one gives 40%, soft voting produces 77.5% → "Up" signal, which correctly follows the confident majority. Hard voting would simply count votes (3 vs. 1 → "Up"), discarding the probability information entirely.

### 11.4 Sequence Length for LSTM

A 30-day sequence length was chosen for the LSTM, corresponding to approximately one calendar month of trading data. This is long enough to capture multi-week trends and reversal patterns, but short enough that the LSTM does not need to learn extremely long-range dependencies that RNNs struggle with. Longer sequences would require more memory and more training time with diminishing returns.

### 11.5 Prophet Settings

Daily seasonality is disabled because the input is already aggregated to daily bars — there is no within-day variation to model. Weekly seasonality is enabled because markets exhibit day-of-week effects (e.g., the Monday effect). Yearly seasonality captures broader patterns like seasonal earnings cycles and tax-loss harvesting in December.

### 11.6 SHAP as an Optional Dependency

SHAP is listed in `requirements.txt` but gracefully skipped if unavailable:

```python
try:
    import shap
except ImportError:
    print("SHAP not available — skipping.")
    return
```

This makes SHAP a progressive enhancement rather than a hard requirement, which is good practice for dependencies that can be complex to install (SHAP requires specific versions of XGBoost and scikit-learn).

---

## 12. Challenges Encountered

### 12.1 Data Leakage Risk

The most persistent challenge in any financial ML project is avoiding data leakage. In this project, several potential leakage points had to be identified and addressed:

- **Feature scaling**: the StandardScaler must be fit only on training data. Fitting on all data would leak test distribution information.
- **Target construction**: `shift(-1)` produces the next day's direction. If any feature construction also inadvertently looks one day forward, it would directly leak the label into the features. All features in `features.py` use only backward-looking operations.
- **Cross-validation**: `TimeSeriesSplit` must be used instead of standard K-fold to prevent future data appearing in earlier validation folds.

### 12.2 NaN Propagation from Feature Engineering

Rolling window computations produce NaN values at the start of each series. With a 50-day moving average, 50 initial rows are invalid. With a 20-day Bollinger Band, 20 rows are invalid. The feature engineering function must drop these cleanly. Missing this step causes silent downstream errors in scikit-learn (which raises errors on NaN inputs) or produces garbage predictions.

### 12.3 LSTM Sequence Alignment with Test Data

For the LSTM, sequences must be created from the test set before evaluation. The test set has `n_test` rows, but after applying `create_sequences` with a 30-day window, it has `n_test - 30 + 1` valid sequences. The corresponding labels must also be trimmed to match. Getting this alignment wrong produces shape mismatches or, worse, silently misaligned predictions that produce misleading metrics.

### 12.4 Prophet Timezone Issues

Prophet internally works with timezone-naive timestamps. If the stock data downloaded from `yfinance` has a timezone-aware DatetimeIndex, Prophet will raise an error during `fit`. The code handles this by passing the index directly to Prophet's `ds` column; if timezone issues arise, `tz_convert(None)` or `tz_localize(None)` can normalise the timestamps.

### 12.5 TensorFlow Logging Verbosity

TensorFlow by default produces extensive logging output (device placement notices, XLA compilation warnings, etc.) that would drown the pipeline's own print statements. The `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"` setting suppresses all but fatal TensorFlow messages. The `verbose=0` argument to `model.fit()` suppresses epoch-by-epoch progress bars. Together these keep the pipeline's console output clean and focused.

---

## 13. Results and Observations

The output directory contains 18 plots covering all three tickers:

**Equity curves** (`equity_AAPL.png`, `equity_JPM.png`, `equity_TSLA.png`): Show the ML strategy equity versus buy-and-hold over the test period, with the drawdown panel below.

**Feature importance charts** (`feature_importance_RandomForest.png`, `feature_importance_XGBoost.png`): Both tree-based models tend to rank recent lagged close prices and momentum features highly, reflecting the short-term autoregressive structure of financial returns.

**SHAP summaries** (`shap_RandomForest.png`, `shap_XGBoost.png`): The SHAP plots reveal directionality — e.g., high RSI values push the model toward predicting "Up" in some configurations, consistent with momentum-following behaviour.

**Learning curves** (6 plots): MLP learning curves typically show training and validation accuracy converging within 30–40 epochs. LSTM curves are generally noisier due to the smaller effective training set size after sequence creation. If validation accuracy diverges upward from training accuracy, it may indicate the dropout rate is too high.

**Prophet forecasts** (`prophet_AAPL.png`, `prophet_JPM.png`, `prophet_TSLA.png`): TSLA shows the widest confidence intervals, reflecting its higher historical volatility. AAPL and JPM produce smoother forecasts with tighter bands. The 180-day forecast horizon is long enough for the uncertainty bands to widen substantially, which is an honest reflection of forecast difficulty.

**Predictions vs. actuals** (3 plots): The scatter of up/down triangle markers over the price series allows visual inspection of whether the model tends to predict the right direction during trending vs. sideways markets.

---

## 14. Reflections on the ML Models

### 14.1 Can Markets Really Be Predicted?

The efficient market hypothesis (EMH) in its weak form asserts that all historical price information is already incorporated into current prices, making technical analysis worthless. If EMH holds perfectly, no feature derived from price history should have predictive power, and all classifiers should perform at the 50% random-baseline level.

In practice, evidence of weak inefficiencies does exist — particularly at high frequencies and in specific anomalies (momentum, value, quality factors). The technical features in this pipeline are exactly the kind of information that EMH says should be useless, yet real-world academic research has shown that momentum features (like the ones included here) do exhibit statistical predictive power over short horizons, though the effect is typically small.

The pipeline is honest about this: the best models will achieve perhaps 52–56% accuracy on direction prediction. This is not impressive in absolute terms but can still be economically meaningful if the model is correct on larger moves.

### 14.2 Why LSTM Doesn't Always Win

It is tempting to assume that the most complex model (LSTM) will always outperform simpler ones. In practice, for tabular financial data at daily granularity, this is often not the case. Several factors work against LSTMs here:

1. **Data quantity**: LSTMs benefit from very large datasets. Ten years of daily data is only ~2500 rows, which is quite small by deep learning standards.
2. **Feature pre-processing already captures temporal patterns**: by the time the feature matrix is built (with lagged prices, rolling volatility, MACD), much of the temporal structure has already been encoded into static features. The LSTM's ability to learn sequence patterns may be redundant.
3. **Overfitting risk**: with 30 × n_features input dimensions and 64 LSTM units plus fully connected layers, the LSTM has many more parameters than the dataset strictly warrants, leading to overfitting even with dropout.

XGBoost and Random Forest are often the most competitive classifiers because tree ensembles are inherently resistant to feature collinearity (which is high here — MA_10, MA_20, MA_50 are all highly correlated) and have built-in regularisation through ensemble averaging.

### 14.3 The Soft-Voting Ensemble's Role

The ensemble's strength is stability. While individual classifiers may have idiosyncratic weaknesses (SVM may overfit on certain sub-periods, XGBoost may underfit on certain regimes), averaging their probabilities smooths out these idiosyncrasies. The ensemble typically performs at or near the best individual classifier, rarely being the worst.

---

## 15. Future Work

### 15.1 Additional Features

The current feature set is entirely price-based. Significant improvements might come from:
- **Fundamental data**: P/E ratios, earnings growth, revenue trends.
- **Sentiment data**: news headline sentiment scores, social media sentiment (e.g., Reddit, Twitter).
- **Options market data**: implied volatility surface, put/call ratios.
- **Cross-asset features**: bond yields (particularly the 10-year US Treasury), USD index, commodity prices.

### 15.2 More Sophisticated Models

- **Transformer-based models**: Temporal Fusion Transformers and similar architectures have demonstrated state-of-the-art performance on structured time-series data.
- **Attention mechanisms**: adding attention layers to the LSTM to allow the model to focus on the most relevant past time steps.
- **Graph neural networks**: capturing inter-stock dependencies (AAPL and SPY, TSLA sector correlations) through graph-structured data.

### 15.3 Walk-Forward Validation

The current approach uses a single train/test split. A more robust evaluation would use walk-forward optimisation: repeatedly re-train the model on all data up to time T, predict on [T, T+window], advance T by the window size, and repeat. This simulates realistic deployment more faithfully and provides confidence intervals on performance metrics.

### 15.4 Risk-Adjusted Position Sizing

The current strategy allocates either 100% or 0% of capital depending on the signal. More sophisticated strategies use the model's predicted probability to size positions proportionally: high-confidence predictions receive larger allocations. Kelly criterion-based sizing is a theoretically optimal approach.

### 15.5 Transaction Cost Modelling

Adding realistic transaction costs (e.g., 5 basis points per trade, representing a realistic brokerage commission plus spread) would substantially change the strategy's profitability, particularly for models that generate many signals (high turnover). This would reveal whether the edge is real or absorbed by friction.

### 15.6 Multi-Asset Portfolio Optimisation

Rather than treating each stock independently, a multi-asset view could use mean-variance optimisation or risk-parity weighting to allocate capital across AAPL, JPM, and TSLA simultaneously, exploiting the diversification benefit of their imperfect return correlation.

### 15.7 Model Monitoring and Drift Detection

In production, model performance degrades as market regimes change. Implementing performance monitoring with alerts when accuracy drops below a threshold — and automatic retraining triggers — would be necessary for a live deployment.

---

## 16. Conclusion

The O4 Stock Predictor is a complete, end-to-end machine learning pipeline that takes a principled approach to a notoriously difficult problem. Its design reflects several important software engineering values:

**Modularity**: each concern (data, features, models, training, evaluation, strategy) is isolated in its own module, making each component independently testable and replaceable.

**Temporal honesty**: all data splitting, cross-validation, and scaling is strictly time-ordered, eliminating the risk of data leakage that would produce misleadingly optimistic performance estimates.

**Model diversity**: by training seven models ranging from logistic regression to LSTM, the pipeline illuminates the performance landscape and prevents the common mistake of picking the most complex model by default.

**Visualisation depth**: the 18+ output plots cover feature importance, SHAP explainability, learning curves, forecast uncertainty, prediction quality, and trading performance — providing a multi-faceted view of what the models have learned and how they behave.

**Test coverage**: unit tests validate the contracts of every major function, ensuring that refactoring or extending the codebase does not silently break existing behaviour.

The project demonstrates that a clean, well-engineered ML pipeline is achievable even in a domain as complex and noisy as financial markets — and that the engineering discipline around data integrity, evaluation methodology, and uncertainty quantification is at least as important as the choice of model architecture.

Whether any of the models "beat the market" in a practically meaningful way depends on the test period and the specific stocks. What the project demonstrates unambiguously is that building a rigorous baseline is the necessary first step before any more sophisticated approaches are worth attempting.

---

*Journal compiled: May 13, 2026*  
*Project location: `O4/O4---stock-predictor/`*  
*Total source files: 7 Python modules + 4 test files*  
*Total output artefacts: 18 plots*  
*Dependencies: numpy, pandas, matplotlib, scikit-learn, xgboost, tensorflow, prophet, yfinance, shap*
