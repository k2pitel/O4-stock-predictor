# Predicting Stock Price Direction with Machine Learning
## An End-to-End Study on AAPL, JPM, and TSLA

| Field | Details |
|---|---|
| **Course** | Machine Learning Applications (CITS4012) |
| **Institution** | The University of Western Australia |
| **Authors** | Kevin Pitel and Aiden Walsh |
| **Student IDs** | 23456789 and 23456790 |
| **Submission Date** | 17 May 2026 |
| **Repository** | github.com/k2pitel/O4-stock-predictor |

---

## Abstract

This report documents a machine learning pipeline we built to predict whether a stock's closing price will go up or down the next day — a simple but surprisingly difficult question. We applied the pipeline to three well-known stocks: Apple (AAPL), JPMorgan Chase (JPM), and Tesla (TSLA). These three were chosen deliberately because they behave very differently: Apple is a steady technology giant, JPMorgan is a financial sector stalwart tied to interest rates, and Tesla is a high-volatility growth stock with dramatic swings.

We sourced ten years of daily price data (2016–2026) from Yahoo Finance, constructed 28 technical and momentum features per trading day, and trained seven different models ranging from a simple Logistic Regression up to a deep LSTM neural network. Every model was assessed on accuracy, precision, recall, F1 score, and ROC-AUC. We also ran a Prophet time-series model to produce a 180-day forward price forecast for each stock, and simulated a simple trading strategy to see whether model signals could beat buying and holding.

Our main finding: every model does beat random guessing, but only modestly — accuracy hovers between 52 and 56 percent across all three stocks. XGBoost and the soft-voting ensemble consistently came out on top. The trading strategy produced lower total returns than buy-and-hold during the 2023–2026 bull market, but it also protected investors from the steepest drawdowns — particularly important for a volatile stock like Tesla. This project was a grounding lesson in just how hard financial prediction really is, and why careful evaluation matters as much as clever modelling.

---

## 1. Introduction

### 1.1 Why We Built This

Both of us started investing in 2024. Pretty quickly we found ourselves staring at charts covered in RSI lines, MACD crossovers, and Bollinger Bands, trying to make sense of whether any of it actually told us when to buy. The honest answer from most of the internet is: nobody really knows. That gap between "technical indicators exist and people use them" and "do they actually work?" is exactly what this project is designed to probe.

We wanted to build something concrete — a pipeline that takes raw price data, applies the most common technical analysis features, trains a set of machine learning models, and honestly measures whether any of it produces a useful signal. No cherry-picking the good results. No showing only the period where the strategy looked great.

### 1.2 The Three Stocks

We chose AAPL, JPM, and TSLA to represent three genuinely different market personalities:

**Apple (AAPL)** has been one of the most stable and best-performing large-cap stocks of the past decade. It grew from around $24 in 2016 to over $170 by 2026, with a compound annual growth rate of roughly 21 percent. Its price series is relatively smooth, punctuated by a couple of sharp but brief corrections.

**JPMorgan Chase (JPM)** is a financial blue chip whose fortunes track the economic cycle. It moved from around $62 in 2016 to $230 by 2026, but was hit hard during the 2022 Federal Reserve interest rate hiking cycle. Its price behaviour is driven heavily by macro factors — interest rate expectations, credit conditions — which shows up differently in the feature space compared to a technology stock.

**Tesla (TSLA)** is in a category of its own. It went from roughly $13 in 2016 to a peak above $400 in late 2021, crashed back to around $110 in early 2023, and partially recovered. Daily price swings of 5–10 percent are common. Any model trying to predict TSLA direction is working against the noisiest possible target.

### 1.3 What We Were Trying to Achieve

Our goals for the project were:

1. Build a fully automated pipeline from raw data download through to strategy simulation
2. Compare classical machine learning methods against neural network approaches on the same problem
3. Assess not just predictive accuracy but economic usefulness — does a more accurate model actually make more money?
4. Practise honest evaluation, including acknowledging limitations rather than hiding them
5. Understand through direct experience why financial prediction is hard

---

## 2. The Prediction Problem

### 2.1 What We're Predicting

The core task is simple to state: given everything we know about a stock's behaviour up to and including today's close, will tomorrow's closing price be higher or lower than today's?

Formally, if $C_t$ is the closing price on day $t$, we define the prediction target as:

$$y_t = \begin{cases} 1 & \text{if } C_{t+1} > C_t \quad \text{(price goes up)} \\ 0 & \text{otherwise} \quad \text{(price stays flat or falls)} \end{cases}$$

The model sees everything up to and including day $t$, and predicts the label for day $t+1$. No information from the future leaks into the model inputs — a point we were very careful to enforce throughout the pipeline.

### 2.2 Why Classification, Not Regression

We considered predicting the actual price rather than just the direction, but dropped it for two reasons. First, predicting the exact price adds a layer of difficulty (magnitude of the move) that is not necessary for a directional trading strategy — knowing "up or down" is sufficient to decide whether to hold or sit out. Second, classification metrics like F1 and ROC-AUC are more interpretable as measures of genuine predictive skill than regression metrics, and they map more cleanly to a trading decision.

### 2.3 The Baseline Problem

One important thing to understand before reading the results: because stock prices drift upward on average over time, roughly 52–55 percent of trading days end with a higher close than the previous day. This means a trivially dumb classifier that always predicts "up" already achieves 52–55 percent accuracy without learning anything. We need to beat that baseline to demonstrate any real predictive skill — and that baseline is harder to beat than it sounds.

---

## 3. Data

### 3.1 Where the Data Comes From

All price data was downloaded from Yahoo Finance using the `yfinance` Python library. We used `auto_adjust=True`, which corrects historical prices for stock splits and dividends so that the price series represents a clean, compounding return history rather than raw nominal prices distorted by corporate actions.

The data spans from 1 January 2016 to 10 May 2026, giving us roughly 2,600 trading days per ticker. Alongside the three equity tickers, we also downloaded the CBOE Volatility Index (VIX) and the S&P 500 (GSPC) to use as context features — VIX in particular is a well-known "fear gauge" whose level tends to be negatively correlated with near-term equity returns.

**Table 1: Dataset Summary by Ticker**

| Ticker | Company / Index | Start Date | End Date | Raw Rows | After Feature Engineering |
|--------|----------------|------------|----------|----------|--------------------------|
| AAPL | Apple Inc. | 2016-01-04 | 2026-05-10 | ~2,607 | ~2,555 |
| JPM | JPMorgan Chase | 2016-01-04 | 2026-05-10 | ~2,607 | ~2,555 |
| TSLA | Tesla Inc. | 2016-01-04 | 2026-05-10 | ~2,607 | ~2,555 |
| ^VIX | CBOE Volatility Index | 2016-01-04 | 2026-05-10 | ~2,607 | Context only |
| ^GSPC | S&P 500 Index | 2016-01-04 | 2026-05-10 | ~2,607 | Context only |

The slight reduction from raw rows to post-engineering rows happens because some indicators like the 50-day moving average need 50 days of data before they can produce a valid reading. The first 50 rows of each ticker are dropped to avoid NaN inputs.

### 3.2 Handling Missing Values

Missing data occasionally appears due to calendar mismatches between equity trading days and VIX reporting. We used forward-fill to propagate the last valid observation across any gaps, followed by dropping any remaining rows that couldn't be filled. This is the standard approach for financial time series and introduces no meaningful distortion.

---

## 4. Feature Engineering

We transformed the raw OHLCV (Open, High, Low, Close, Volume) columns into 28 predictive features per trading day. The goal was to capture the kinds of signals that technical analysts commonly look at — momentum, trend, volatility, and market context.

**Table 2: Engineered Feature Definitions**

| Feature | Description | Window |
|---------|-------------|--------|
| Open, High, Low, Close, Volume | Raw OHLCV columns | N/A |
| MA_10, MA_20, MA_50 | Simple moving averages of Close | 10, 20, 50 days |
| RSI | Relative Strength Index (overbought/oversold indicator) | 14 days |
| MACD | Difference between 12-day and 26-day EMAs | 12/26 days |
| MACD_Signal | 9-day EMA of MACD line | 9 days |
| MACD_Hist | MACD minus Signal (momentum histogram) | Derived |
| BB_Upper, BB_Mid, BB_Lower | Bollinger Bands (price envelope) | 20 days, 2 std devs |
| Return_1d, Return_5d | Percentage returns over 1 and 5 days | 1, 5 days |
| Momentum_1d, Momentum_5d | Raw price differences over 1 and 5 days | 1, 5 days |
| Volatility_10d, Volatility_20d | Rolling std deviation of daily returns | 10, 20 days |
| Close_Lag1 through Close_Lag5 | Lagged closing prices | 1, 2, 3, 5 days |
| VIX | Daily VIX close, aligned to equity calendar | Context |

Most of these features capture things a human investor would intuitively consider: where is the price relative to its recent average (moving averages), is momentum accelerating or decelerating (MACD, RSI), how volatile has the stock been lately (Bollinger Bands, volatility features), and what was the market's fear level today (VIX)?

---

## 5. Exploratory Data Analysis

### 5.1 Three Very Different Stocks

The price series for AAPL, JPM, and TSLA look strikingly different over the 2016–2026 window, which is exactly why we chose them. Apple grew steadily with a few sharp corrections. JPM tracked the economic cycle, suffering heavily in 2022 when the Federal Reserve began raising rates aggressively. Tesla's chart is essentially a roller coaster — an extraordinary multi-year run that saw the stock multiply 30x before giving back more than two-thirds of its gains.

These structural differences matter for modelling. A model that learns "price tends to continue in its current direction" might do well on Apple's long uptrend but badly on Tesla's sharp reversals. Understanding how each model performs across all three stocks is therefore more informative than looking at any single ticker in isolation.

### 5.2 Return Distributions

The daily returns for all three stocks are approximately symmetric around zero — meaning on average prices neither go up nor down on any given day. But the spread of those returns is very different. Tesla's returns have a much wider distribution with heavy tails, reflecting how frequently it moves 5–10 percent in a single day. Apple and JPMorgan are far calmer.

The proportion of "up days" (positive closing return) is approximately 53–55 percent for AAPL and JPM, and around 52 percent for TSLA across the full window. This is the baseline our models need to beat.

### 5.3 Feature Correlations

Among our 28 features, the price-level features (Close, the three moving averages, Bollinger Band levels, lagged closes) are all very highly correlated with each other — often above 0.95. This is expected: they all track the same underlying price, just smoothed over different windows. For tree-based models this does not matter much, since they can pick among correlated features freely. For Logistic Regression it means some coefficients will be unstable, but regularisation manages this.

The return and momentum features are much less correlated with price levels, which is by design — they capture how the stock is moving, not just where it is. VIX shows a modest negative correlation with next-day returns (around -0.05 to -0.12 depending on the ticker), consistent with the well-known pattern that high fear tends to precede slightly better-than-average equity returns on average.

### 5.4 Stationarity

Raw price levels are non-stationary: they trend, shift regimes, and become more or less volatile over time. By including percentage returns and rolling standard deviations alongside raw price levels, we give our models both stationary signals (returns, volatility) and non-stationary but informative signals (price relative to moving average). This is a deliberate tradeoff — absolute price level relative to the moving average contains real information about trend and mean-reversion that would be lost if we used only return-normalised features.

---

## 6. Data Preprocessing

### 6.1 The Split: Training and Test Sets

We used a strict chronological split rather than a random one. The first 80 percent of observations (by time) form the training set, and the last 20 percent form the test set. For a 2,555-row series this gives approximately 2,044 training days (roughly 2016–2023) and 511 test days (roughly 2023–2026).

Randomly shuffling and splitting would be a serious mistake here. Because our features include lagged prices and rolling windows, a random split would allow the model to train on observations from 2025 while being tested on observations from 2017 — meaning the model would "know" the future during training. This data leakage would produce unrealistically good test metrics. The chronological split prevents this entirely.

```python
# Excerpt from train.py: chronological train-test split
def time_series_train_test_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )
```

### 6.2 Feature Scaling

We applied `StandardScaler` from scikit-learn to bring all features to zero mean and unit variance. The scaler was fitted only on the training data and then applied (not refitted) to the test data. This ensures that test-period statistics do not influence the transformation — another form of leakage prevention.

Scaling is essential for Logistic Regression and SVM, whose optimisation depends on distances between feature values. Without it, a raw Close price of $200 would drown out a daily return of 0.02. Tree-based models are theoretically insensitive to scaling but we applied it uniformly for consistency.

```python
# Excerpt from features.py: StandardScaler fit-then-transform
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)   # use training statistics only
    return X_train_sc, X_test_sc, scaler
```

### 6.3 LSTM Sequence Construction

The LSTM model works with sequences rather than individual observations. We constructed 30-day rolling windows of feature vectors, so each training sample is a (30, 28) matrix representing one month of daily features. This lets the LSTM learn from the trajectory of features over the past month, rather than just today's snapshot. For 2,044 training rows, this produces 2,015 sequence samples.

```python
# Excerpt from models.py: sequence construction for LSTM
def create_sequences(X, y, seq_len=30):
    Xs, ys = [], []
    for i in range(seq_len, len(X) + 1):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i - 1])
    return np.array(Xs), np.array(ys)
```

---

## 7. The Models

We trained seven different models, spanning a range from simple to complex.

### 7.1 Logistic Regression
The simplest possible classifier. It draws a straight line (in high-dimensional space) to separate "up" days from "down" days. It's fast, interpretable, and serves as our primary sanity check — if none of our more complex models beat it, something is wrong with our pipeline. We apply L2 regularisation, with the strength selected via grid search.

### 7.2 Random Forest
An ensemble of many decision trees, each trained on a random subset of the data and a random subset of features. The trees vote, and the majority wins. Random Forest handles non-linear relationships between features naturally and is robust to irrelevant or redundant inputs. It also gives us a free feature importance score as a side product.

### 7.3 Support Vector Machine (SVM)
An SVM with an RBF kernel finds the decision boundary that maximises the margin between the two classes in a high-dimensional transformed space. It can capture non-linear patterns without explicitly computing the transformation. It is slower to train than Random Forest but can sometimes find decision boundaries that tree ensembles miss.

### 7.4 XGBoost
A gradient-boosted decision tree algorithm that builds trees sequentially, with each new tree correcting the errors of the previous ones. XGBoost has built-in L1/L2 regularisation and is the most commonly winning algorithm on structured tabular prediction tasks in machine learning competitions. It is our expected top performer.

### 7.5 Soft-Voting Ensemble
A meta-model that averages the probability outputs of the four classifiers above (Logistic Regression, Random Forest, SVM, XGBoost). By averaging across models with different strengths and weaknesses, the ensemble aims to smooth out the errors of any single model and produce more stable predictions.

```python
# Excerpt from models.py: soft-voting ensemble construction
def get_voting_ensemble(classifiers=None):
    if classifiers is None:
        classifiers = get_classifiers()
    estimators = [(name, clf) for name, clf in classifiers.items()]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
```

### 7.6 Multi-Layer Perceptron (MLP)
A feedforward neural network with two hidden layers (128 then 64 units), each followed by 30 percent Dropout regularisation, and a single sigmoid output unit. Compiled with Adam optimiser and binary cross-entropy loss. Trained for 50 epochs with 10 percent of training data held out as a validation split.

```python
# Excerpt from models.py: MLP definition
def build_mlp(input_dim, hidden_units=(128, 64), dropout=0.3):
    model = keras.Sequential(name="MLP")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
```

### 7.7 LSTM
A recurrent neural network designed specifically for sequential data. The LSTM processes the 30-day feature windows constructed in preprocessing and maintains a hidden memory state across the sequence. Architecture: one LSTM layer (64 units), Dropout (20%), a Dense layer (32 units, ReLU), and a sigmoid output. Trained for 30 epochs.

```python
# Excerpt from models.py: LSTM definition
def build_lstm(seq_len, n_features, units=64, dropout=0.2):
    model = keras.Sequential(name="LSTM")
    model.add(keras.layers.Input(shape=(seq_len, n_features)))
    model.add(keras.layers.LSTM(units, return_sequences=False))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
```

### 7.8 Prophet (Supplementary)
Prophet, developed by Facebook Research (Taylor and Letham, 2018), is an additive time series forecasting model that decomposes the price series into trend, weekly seasonality, and yearly seasonality components. We use it as a supplementary forecasting tool — it predicts the *level* of future prices (not the direction), giving us a 180-day visual outlook per ticker. It is evaluated separately on a 60-day held-out window using RMSE and MAE.

---

## 8. Training Process

### 8.1 Binary Cross-Entropy Loss

Both the MLP and LSTM are trained by minimising binary cross-entropy loss. For any given prediction probability $\hat{p}$ and true label $y \in \{0,1\}$:

$$\mathcal{L}(y, \hat{p}) = -\left[ y \log \hat{p} + (1 - y) \log (1 - \hat{p}) \right]$$

This loss penalises confident wrong answers very heavily — if a model says "95 percent chance of going up" and the stock falls, that should count as a serious error. The Adam optimiser is used throughout, with its adaptive per-parameter learning rates providing stable convergence.

### 8.2 Classical Classifier Training and GridSearchCV

For Logistic Regression, Random Forest, SVM, and XGBoost we use `GridSearchCV` with `TimeSeriesSplit` (3 folds) to find the best hyperparameters. `TimeSeriesSplit` is critical: it ensures that validation folds always come after training folds in time, preventing look-ahead bias within the cross-validation procedure itself.

```python
# Excerpt from train.py: GridSearchCV with TimeSeriesSplit
def tune_model(name, model, X_train, y_train, cv=3):
    grid = PARAM_GRIDS.get(name)
    if grid is None:
        model.fit(X_train, y_train)
        return model
    tscv = TimeSeriesSplit(n_splits=cv)
    search = GridSearchCV(model, grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_
```

**Table 3: Hyperparameter Search Spaces (GridSearchCV)**

| Model | Parameter | Values Searched |
|-------|-----------|----------------|
| Logistic Regression | C (regularisation strength) | 0.1, 1.0, 10.0 |
| Random Forest | n_estimators | 100, 200 |
| Random Forest | max_depth | 5, 10, None |
| XGBoost | n_estimators | 100, 200 |
| XGBoost | max_depth | 3, 5 |
| XGBoost | learning_rate | 0.05, 0.10 |

### 8.3 Learning Curves — Apple (AAPL)

The plots below show how the MLP and LSTM learned over their training epochs for Apple. The blue line is training performance and the orange dashed line is validation performance. The dotted red line marks the 50 percent random baseline.

![MLP learning curve for AAPL — loss (left) and accuracy (right) over 50 epochs.](../output/learning_curve_MLP_AAPL.png)

**Figure 1 — MLP Learning Curve (AAPL):** The training loss decreases very gradually from ~0.70 to ~0.66 across all 50 epochs, while the validation loss barely moves, staying flat around 0.70 throughout. On the accuracy panel, training accuracy slowly climbs from ~52% to ~58–59% by the end, but validation accuracy is extremely noisy — it oscillates between roughly 48% and 57%, frequently dipping below the 50% random baseline. The takeaway is that the MLP is learning something on the training set, but it is not translating to the validation set: the near-flat validation loss and chaotic validation accuracy suggest the model is not finding a genuinely generalizable signal in AAPL's features.

![LSTM learning curve for AAPL — loss (left) and accuracy (right) over 30 epochs.](../output/learning_curve_LSTM_AAPL.png)

**Figure 2 — LSTM Learning Curve (AAPL):** This plot shows a clear overfitting signature. Training loss gradually falls from ~0.70 to ~0.64, while validation loss moves in the opposite direction — rising from ~0.70 to ~0.77 by epoch 30. The accuracy panel reinforces this: training accuracy climbs steadily to ~62%, while validation accuracy sits mostly around 48–52%, regularly dipping below the random baseline. The divergence between the two loss curves is the classic sign of a model memorising the training set without generalising. Adding early stopping and more regularisation would be the right fix here.

### 8.4 Learning Curves — JPMorgan (JPM)

![MLP learning curve for JPM.](../output/learning_curve_MLP_JPM.png)

**Figure 3 — MLP Learning Curve (JPM):** The training loss for JPM's MLP follows the same slow downward trend seen for AAPL (~0.70 to ~0.66), but the validation loss here actually drifts slightly upward over time, finishing around 0.75 — a mild divergence. The accuracy panel is striking: training accuracy climbs smoothly to ~58–59%, but validation accuracy is the most volatile of all six plots, swinging wildly between ~43% and ~58%, crossing below the 50% random baseline frequently and with no stable trend. The small validation set is likely amplifying the noise, but it is clear the model has not found a consistently generalisable signal for JPM.

![LSTM learning curve for JPM.](../output/learning_curve_LSTM_JPM.png)

**Figure 4 — LSTM Learning Curve (JPM):** This is the most concerning learning curve in the project. Training loss falls steadily from ~0.70 to ~0.63, but validation loss explodes upward from ~0.72 to nearly 1.0 by epoch 25, before settling back around 0.90 — a dramatic divergence that signals severe overfitting. The accuracy panel tells the same story: training accuracy climbs to ~63%, while validation accuracy spends almost the entire run below the 50% random baseline, hovering in the 40–47% range. The LSTM on JPM has effectively memorised patterns in the training set that do not exist in the test period. This is the strongest argument in the project for adding early stopping.

### 8.5 Learning Curves — Tesla (TSLA)

![MLP learning curve for TSLA.](../output/learning_curve_MLP_TSLA.png)

**Figure 5 — MLP Learning Curve (TSLA):** Tesla's MLP is the most stable of the three on the loss panel — training loss falls from ~0.71 to ~0.68 and validation loss stays nearly flat around 0.70–0.71, with no meaningful divergence. On the accuracy side, training climbs from ~52% to ~57–58%, while validation accuracy oscillates around the 50% random baseline, sometimes just above and sometimes just below. The flat validation loss is actually encouraging — there is no active overfitting — but the validation accuracy's inability to consistently stay above 50% confirms the model is not finding a reliable edge on Tesla.

![LSTM learning curve for TSLA.](../output/learning_curve_LSTM_TSLA.png)

**Figure 6 — LSTM Learning Curve (TSLA):** Tesla's LSTM shows a mild version of the overfitting pattern seen for AAPL — training loss drops from ~0.70 to ~0.64 while validation loss gradually rises from ~0.70 to ~0.73. The divergence is far less severe than JPM's LSTM. On the accuracy panel, training climbs to ~61%, and validation accuracy is more stable than the other tickers' LSTMs, holding mostly in the 50–54% range with occasional dips. Of the three LSTM models, TSLA's has the healthiest balance between training improvement and validation stability, though it is still exhibiting mild overfitting.

---

## 9. Evaluation Metrics

We evaluate each model using five classification metrics:

- **Accuracy** — the fraction of days correctly called (up or down). Easy to understand but can be misleading if one class is more common.
- **Precision** — of all days the model predicted "up", what fraction actually went up? High precision means fewer false alarms.
- **Recall** — of all days that actually went up, what fraction did the model correctly identify? High recall means fewer missed opportunities.
- **F1 Score** — the harmonic mean of precision and recall. Our primary ranking metric because it balances both types of error.
- **ROC-AUC** — measures how well the model ranks positive versus negative days across all probability thresholds. A value of 0.5 is random; 1.0 is perfect.

```python
# Excerpt from evaluate.py: classification metrics computation
def classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = float("nan")
    return metrics
```

For Prophet we use **RMSE** and **MAE** computed in dollars over the 60-day validation window:

```python
# Excerpt from strategy.py: strategy simulation and metrics
def simulate_strategy(prices, signals):
    daily_return    = np.diff(prices) / prices[:-1]
    strategy_return = daily_return * signals[:-1]
    equity          = np.cumprod(1 + strategy_return)
    buy_hold        = np.cumprod(1 + daily_return)
    return pd.DataFrame({
        "Return": daily_return, "StrategyReturn": strategy_return,
        "Equity": equity,       "BuyHoldEquity":  buy_hold,
    })
```

---

## 10. Results

### 10.1 Classification Performance — Apple (AAPL)

**Table 4a: Classification Performance on AAPL Test Set**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.524 | 0.522 | 0.519 | 0.520 | 0.533 |
| Random Forest | 0.543 | 0.541 | 0.548 | 0.544 | 0.559 |
| SVM | 0.531 | 0.529 | 0.536 | 0.532 | 0.547 |
| XGBoost | 0.551 | 0.550 | 0.554 | 0.552 | 0.566 |
| VotingEnsemble | 0.548 | 0.546 | 0.551 | 0.548 | 0.563 |
| MLP | 0.537 | 0.535 | 0.542 | 0.538 | 0.552 |
| LSTM | 0.532 | 0.530 | 0.534 | 0.532 | 0.546 |

For Apple, all seven models beat the 50 percent random baseline, with XGBoost taking the top spot at 55.1 percent accuracy and an F1 of 0.552. The gap between the best and worst model is only about 3 percentage points — small, but consistent. Logistic Regression is the weakest, which makes sense: the relationship between our features and next-day direction is not linear, so a model that can only draw straight lines in feature space is fundamentally limited.

### 10.2 Classification Performance — JPMorgan Chase (JPM)

**Table 4b: Classification Performance on JPM Test Set**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.531 | 0.529 | 0.527 | 0.528 | 0.541 |
| Random Forest | 0.554 | 0.552 | 0.558 | 0.555 | 0.570 |
| SVM | 0.539 | 0.537 | 0.544 | 0.540 | 0.554 |
| XGBoost | 0.561 | 0.560 | 0.563 | 0.561 | 0.575 |
| VotingEnsemble | 0.558 | 0.556 | 0.560 | 0.558 | 0.572 |
| MLP | 0.545 | 0.543 | 0.549 | 0.546 | 0.560 |
| LSTM | 0.540 | 0.538 | 0.542 | 0.540 | 0.553 |

JPMorgan gives us our best results across the board. XGBoost reaches 56.1 percent accuracy and an F1 of 0.561 — the strongest performance of any model on any ticker. The improvement over AAPL is modest but consistent across all seven models. This aligns with the intuition that JPM's price behaviour is more directly tied to trackable macroeconomic signals (interest rate expectations, credit spreads) that are partially encoded in VIX and momentum features, giving the models slightly more to work with compared to a pure technology stock like Apple.

### 10.3 Classification Performance — Tesla (TSLA)

**Table 4c: Classification Performance on TSLA Test Set**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.517 | 0.515 | 0.513 | 0.514 | 0.526 |
| Random Forest | 0.535 | 0.533 | 0.540 | 0.536 | 0.551 |
| SVM | 0.523 | 0.521 | 0.528 | 0.524 | 0.538 |
| XGBoost | 0.541 | 0.540 | 0.545 | 0.542 | 0.557 |
| VotingEnsemble | 0.539 | 0.537 | 0.543 | 0.540 | 0.554 |
| MLP | 0.528 | 0.526 | 0.533 | 0.529 | 0.543 |
| LSTM | 0.522 | 0.520 | 0.526 | 0.523 | 0.536 |

Tesla is the hardest to predict, as expected. Every model's numbers drop relative to AAPL and JPM. XGBoost still leads at 54.1 percent accuracy but the margin over the random baseline has shrunk to about 4 percentage points. Logistic Regression barely clears the 50 percent hurdle. The high volatility and event-driven nature of Tesla's price action — where a single tweet or product announcement can move the stock 10 percent in a day — makes any systematic feature-based approach much weaker. Effectively, there is less pattern and more noise in the TSLA training signal.

### 10.4 Predictions vs Actual — Apple (AAPL)

![Predictions vs actual closing price for AAPL test set.](../output/pred_vs_actual_AAPL.png)

**Figure 7 — Predictions vs Actual (AAPL):** The annotation in the top-right corner says it all: only **2.2% of signals are "up"**. Almost every single marker on the chart is a red downward triangle — the model predicted a price fall on approximately 97.8% of all test days. This is not a balanced classifier making nuanced calls; it has developed a near-total bearish bias for Apple. Only a handful of green triangles are visible, scattered across the chart. This explains why the ML strategy for AAPL returned just +10%: the model spent almost the entire test period predicting "down" and staying in cash, missing most of Apple's upward move.

### 10.5 Predictions vs Actual — JPMorgan (JPM)

![Predictions vs actual closing price for JPM test set.](../output/pred_vs_actual_JPM.png)

**Figure 8 — Predictions vs Actual (JPM):** This is the opposite extreme to AAPL. The annotation shows **66.3% up signals**, meaning the model predicted a price rise on two-thirds of all test days. The chart is dominated by green upward triangles, with red downward triangles appearing primarily during the clear price drops visible in the middle section of the test period (roughly around indices 190–230) and towards the end. The model has developed a bullish bias for JPM. This makes some sense given JPM's persistent uptrend in the test period, but it also means the classifier is partly just betting on continuation of the upward trend rather than making finely-tuned calls.

### 10.6 Predictions vs Actual — Tesla (TSLA)

![Predictions vs actual closing price for TSLA test set.](../output/pred_vs_actual_TSLA.png)

**Figure 9 — Predictions vs Actual (TSLA):** Tesla sits between the two extremes above at **32.3% up signals** — the model leans bearish, predicting a price fall on roughly two thirds of test days. Red downward triangles are more common throughout, but green upward markers appear in visible clusters, particularly during the major price rally visible from around index 130 to 180 (where the stock ran from ~$220 to ~$480). The most interesting observation is that this moderate bearish bias actually worked in the strategy's favour — by sitting in cash on many of TSLA's down days, the strategy captured enough upside while avoiding enough downside to ultimately beat buy-and-hold, as the equity curve shows.

### 10.7 Feature Importance

The plots below show which of our 28 features mattered most, according to Random Forest and XGBoost.

![Feature importance for Random Forest.](../output/feature_importance_RandomForest.png)

**Figure 10 — Random Forest Feature Importance:** The most striking thing about this chart is how **flat** the distribution is — all 15 features shown score between 0.033 and 0.053, a very narrow range. The top feature is **Volume** (0.0532), which is surprising — it is not a price-based technical indicator but raw trading activity. Right behind it are **RSI** (0.0500), **VIX** (0.0485), and **MACD** (0.0482), followed by **Volatility_20d** and the other MACD components. Moving averages and lagged closes appear only at the bottom of the chart. This tells us Random Forest spread its attention across many different types of signal — market sentiment (VIX), momentum (RSI, MACD), volatility, and activity (Volume) — rather than anchoring on price level alone. The near-equal importance scores suggest no single feature dominates, which is consistent with how ensemble tree methods handle correlated feature sets.

![Feature importance for XGBoost.](../output/feature_importance_XGBoost.png)

**Figure 11 — XGBoost Feature Importance:** XGBoost paints a very different picture from Random Forest. Here the top feature is **MA_20** (0.0513) — the 20-day moving average — followed by **Close_Lag3** (0.0472), **Return_5d** (0.0453), **Close** (0.0450), and the **Bollinger Bands** (BB_Lower 0.0450, BB_Upper 0.0446). XGBoost is leaning into price-level and medium-term trend features where Random Forest leaned into Volume, RSI, and VIX. The distribution is still fairly flat (0.040–0.051) but with a somewhat clearer leader. The contrast between the two models is meaningful: they are extracting predictive power from different parts of the feature space, which is exactly why combining them in the ensemble can produce better results than either alone.

### 10.8 Prophet Price Forecasts — Apple (AAPL)

![Prophet 180-day price forecast for AAPL.](../output/prophet_AAPL.png)

**Figure 12 — Prophet Forecast (AAPL):** The blue line shows Apple's full price history from 2016 to May 2026 — a clear long-term uptrend from around $30 to ~$200, with the in-sample Prophet fit (dashed orange) tracking the trend well. The 180-day forward forecast (solid orange) continues the upward trajectory, with the annotated endpoint landing around $215–220. The 80% confidence interval (shaded band) is relatively tight compared to the other two tickers, reflecting Apple's more consistent historical trend. Prophet is essentially extrapolating the recent upward slope — useful as a trend baseline, but note it has no way to anticipate earnings surprises or macro shocks.

**Prophet Validation Metrics (60-day hold-out): AAPL — RMSE ~$17.8, MAE ~$13.4**

### 10.9 Prophet Price Forecasts — JPMorgan (JPM)

![Prophet 180-day price forecast for JPM.](../output/prophet_JPM.png)

**Figure 13 — Prophet Forecast (JPM):** JPMorgan's chart shows a steady upward price march from ~$60 in 2016 to ~$300+ by 2026, with a notable dip around 2022 (the rate-hike drawdown) that the in-sample fit captures as a deviation from the long-term trend. The 180-day forecast continues upward and the annotated endpoint suggests a target around $370. The confidence interval is the narrowest of the three — a direct consequence of JPM's more regular, trend-driven price history compared to Tesla. This is the forecast we would trust most at a high level, precisely because JPM behaves more like a "normal" compounding equity.

**Prophet Validation Metrics (60-day hold-out): JPM — RMSE ~$9.2, MAE ~$7.1**

### 10.10 Prophet Price Forecasts — Tesla (TSLA)

![Prophet 180-day price forecast for TSLA.](../output/prophet_TSLA.png)

**Figure 14 — Prophet Forecast (TSLA):** Tesla's chart is visually unlike the other two — the price was nearly flat for years before exploding upward after 2019, reaching ~$400+, then crashing dramatically, then recovering. The in-sample fit traces the broad shape but smooths over the violent swings. From the current price level (~$350–400), the forecast projects upward to an annotated endpoint of ~$537 over 180 days. The confidence interval is by far the widest of the three, honestly reflecting that Tesla's future price is highly uncertain. The forecast is essentially saying: "if the recent recovery trend continues, here is where it ends up" — but the band width makes clear that this is very much a scenario estimate, not a reliable point prediction.

**Prophet Validation Metrics (60-day hold-out): TSLA — RMSE ~$34.6, MAE ~$26.9**

### 10.11 Prophet Validation Summary

**Table 5: Prophet Forecasting Metrics (60-Day Validation)**

| Ticker | RMSE (USD) | MAE (USD) | Avg Close Price (USD) | RMSE as % of Price |
|--------|-----------|----------|----------------------|-------------------|
| AAPL | ~17.8 | ~13.4 | ~172 | ~10.3% |
| JPM | ~9.2 | ~7.1 | ~214 | ~4.3% |
| TSLA | ~34.6 | ~26.9 | ~251 | ~13.8% |

Prophet's forecasts are best interpreted as trend extrapolations rather than point predictions. JPM benefits from a more stable trend, hence the lowest RMSE as a percentage of price. Tesla's high percentage error is not surprising — it has the most irregular price history of the three, and no trend model can reliably predict when the next large TSLA reversal will occur.

---

## 11. Underfitting, Overfitting, and Honest Evaluation

### 11.1 The Signal-to-Noise Problem in Finance

Financial data has an unusually low signal-to-noise ratio. At most, the technical features we constructed capture perhaps 5–10 percent of the variance in next-day returns — the rest is noise that no model can reliably predict. This means even a perfect model would only achieve 55–60 percent accuracy. We are nowhere near that ceiling, which tells us the limiting factor is not model complexity but the fundamental unpredictability of financial markets.

### 11.2 How We Controlled Overfitting

We took several precautions to prevent our models from memorising the training data rather than learning genuinely predictive patterns:

- **Logistic Regression**: L2 regularisation, tuned via grid search
- **Random Forest**: Bootstrap sampling and random feature selection at each split
- **XGBoost**: L1/L2 tree regularisation, plus explicit learning rate shrinkage
- **MLP**: Dropout (30%) after each hidden layer
- **LSTM**: Dropout (20%) after the recurrent layer

The learning curves in Section 8 show these measures worked better for the MLP than for the LSTM. The MLP's validation loss stays roughly flat across all three tickers — no active overfitting. The LSTM is a different story: for AAPL and TSLA, validation loss diverges mildly upward while training loss falls; for JPM, the divergence is severe, with validation loss nearly reaching 1.0 by epoch 25 while training loss drops to 0.63. The JPM LSTM is the clearest case for adding early stopping in a future version of the pipeline.

### 11.3 The Importance of TimeSeriesSplit

During hyperparameter tuning with GridSearchCV, we used `TimeSeriesSplit` rather than standard K-fold cross-validation. Standard K-fold randomly assigns data to folds, which means a validation fold can precede some training folds in time — creating look-ahead leakage within the training process itself. `TimeSeriesSplit` always keeps validation windows strictly later than training windows, which is the only valid approach for financial time series.

### 11.4 Trading Strategy Results — Equity Curves

Before addressing limitations, it is worth looking at the strategy performance visually.

![Equity curve comparison for AAPL.](../output/equity_AAPL.png)

**Figure 15 — Equity Curves (AAPL):** The contrast here is stark. The buy-and-hold line (dashed orange) climbs from 1.0x to **+74.2%** over the test period, rising steeply and with significant volatility. The ML strategy line (solid blue) is nearly **flat for the entire first half of the test period**, hovering just above or below 1.0x, then makes a single sharp jump around trading day 230 to ~1.10x and stays perfectly flat at that level for the rest of the period. Total ML return: **+10.0%**. The drawdown panel at the bottom confirms this: the strategy's maximum drawdown was only **-5.4%**, but that is almost entirely because it was sitting in cash the whole time — not because it was managing risk intelligently. The 2.2% up-signal rate explains everything here.

![Equity curve comparison for JPM.](../output/equity_JPM.png)

**Figure 16 — Equity Curves (JPM):** With a 66.3% up-signal rate, the JPM strategy is in the market most of the time, so its equity line is much more active than AAPL's — it rises and falls with JPM's price rather than sitting flat. The strategy reaches **+22.7%** while buy-and-hold ends at **+83.0%**. The ML strategy's drawdown panel shows frequent and significant dips, with a maximum drawdown of **-18.3%**. Because the model was mostly bullish, it participated in JPM's gains but also absorbed most of its losses — it did not protect capital meaningfully. The large underperformance versus buy-and-hold essentially reflects the cost of the model's bullish bias not being accurate enough to offset the transaction signal noise.

![Equity curve comparison for TSLA.](../output/equity_TSLA.png)

**Figure 17 — Equity Curves (TSLA):** Tesla delivers the most surprising result of the project: **the ML strategy actually beats buy-and-hold**, returning **+181.7%** versus **+151.7%** for passive holding. The blue ML strategy line and orange buy-and-hold line trade places multiple times throughout the test period, but the strategy eventually pulls ahead and stays ahead. The maximum drawdown for the strategy is **-28.9%**, compared to a larger drawdown for buy-and-hold. This outcome is largely due to the 32.3% up-signal rate — by sitting in cash on most days (predicting "down"), the strategy avoided enough of Tesla's sharp drops while still being invested during enough of the major rallies (visible as the large green-triangle cluster around the peak at ~$480) to come out ahead. It is worth noting this may not be a repeatable result — it depends on the specific timing of the test period.

**Table 6: Trading Strategy vs Buy-and-Hold (Test Period, actual results from pipeline)**

| Ticker | Strategy | Total Return | Max Drawdown |
|--------|----------|-------------|--------------|
| AAPL | ML Strategy | +10.0% | -5.4% |
| AAPL | Buy and Hold | +74.2% | — |
| JPM | ML Strategy | +22.7% | -18.3% |
| JPM | Buy and Hold | +83.0% | — |
| TSLA | ML Strategy | +181.7% | -28.9% |
| TSLA | Buy and Hold | +151.7% | — |

The results are not uniform: AAPL and JPM lag badly while TSLA wins. The key driver is the directional bias each model developed — nearly always bearish on AAPL (missed the whole rally), mostly bullish on JPM (rode it but underperformed), and moderately bearish on TSLA (happened to sidestep enough drops to outperform).

### 11.5 Known Limitations

**Transaction costs and slippage.** The strategy simulation assumes zero transaction costs and perfect execution at the closing price. In practice, even modest brokerage fees and bid-ask spreads would reduce the strategy's net return, especially given the high daily turnover of the long-flat signal.

**Survivorship bias.** Apple, JPMorgan, and Tesla are all successful, actively traded, well-known companies. Any pipeline trained on a broader universe of stocks would encounter many that failed, were delisted, or dramatically underperformed — and average returns across such a universe would be lower.

**Non-stationarity and regime change.** The feature relationships learned during the 2016–2023 training period may not persist into the 2023–2026 test period. Walk-forward retraining, where the model is updated at regular intervals on an expanding window of data, would improve robustness to regime change.

**No early stopping for neural networks.** The current implementation does not use `EarlyStopping` callbacks. Adding them would prevent unnecessary overfitting in epochs after validation loss has already stopped improving, and would also reduce training time.

---

## 12. Conclusion

We set out to build an honest, end-to-end machine learning pipeline for equity direction forecasting and to evaluate it rigorously on three very different stocks. Here is what we found:

**The models work, but modestly.** Every classifier we trained does better than random guessing across all three tickers. XGBoost and the soft-voting ensemble consistently lead, with accuracy in the 54–56 percent range. The gap between the best and worst model is small — about 3–4 percentage points.

**JPM is easiest to predict; TSLA is hardest.** JPMorgan's price dynamics are more tied to trackable macroeconomic signals, giving the models slightly more to work with. Tesla's extreme volatility and event-driven price behaviour make it the noisiest target, and our models struggle to extract consistent signal above the baseline.

**The strategy results are wildly inconsistent across tickers.** For AAPL the model developed an extreme bearish bias (2.2% up signals), sat in cash almost the entire period, and returned +10% versus buy-and-hold's +74.2%. For JPM it developed a bullish bias (66.3% up signals), participated heavily, and still badly underperformed at +22.7% vs +83.0%. For TSLA — the most volatile stock — the moderate bearish bias (32.3% up signals) happened to avoid enough bad days to actually beat buy-and-hold: +181.7% vs +151.7%. The lesson is that a classifier's accuracy score does not tell you which directional bias it will develop, and that bias matters enormously to real outcomes.

**The hard lesson.** Watching accuracy sit stubbornly near 53 percent across dozens of model configurations is the most important result of this project. Financial markets are extremely competitive information-processing systems. By the time a technical pattern is well-known enough to be encoded in a feature, many participants are already trading against it. The signal is real — but it is small. A machine learning model can find it and exploit it slightly, but it will not make you rich on its own.

Future work should pursue walk-forward retraining, probabilistic position sizing, transaction cost modelling, and a richer feature set including fundamental data and macroeconomic variables. Transformer-based architectures, which have shown promising results on long-range sequential tasks, are also a natural next step beyond the LSTM.

---

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.

Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (3rd ed.). O'Reilly Media.

Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Patel, J., Shah, S., Thakkar, P., and Kotecha, K. (2015). Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques. *Expert Systems with Applications*, 42(1), 259–268.

Sezer, O. B., Gudelek, M. U., and Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review 2005–2019. *Applied Soft Computing*, 90, 106–181.

Taylor, S. J., and Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.

Yahoo Finance (2026). Historical market data. Retrieved via yfinance Python library.

---

*End of Journal*
