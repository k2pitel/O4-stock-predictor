# O4 Report: End-to-End Machine Learning for Stock Market Direction Prediction and Trading Strategy Simulation

---

**Course:** SWMAL-01 Machine Learning
**University:** Technical University of Denmark (DTU)
**Group:** Gruppe 29
**Members:** Kevin
**Submission Date:** May 17, 2026

---

## Abstract

This report documents an end-to-end machine learning project aimed at predicting the next-day directional movement of three major United States equities: Apple Inc. (AAPL), JPMorgan Chase (JPM), and Tesla Inc. (TSLA). The dataset was sourced from Yahoo Finance via the `yfinance` Python library and spans the period from January 2016 to May 2026, providing approximately 2,500 trading days per ticker after preprocessing. A rich feature set was engineered from the raw OHLCV time series, encompassing classical technical indicators including the Relative Strength Index, the Moving Average Convergence Divergence indicator, Bollinger Bands, simple moving averages, exponential moving averages, rolling volatility estimates, lagged close prices, and the CBOE Volatility Index as a cross-asset sentiment signal.

The prediction problem was framed as binary classification, with the target variable indicating whether the next trading day's closing price would exceed the current day's close. Seven machine learning architectures were trained and compared: Logistic Regression, Random Forest, Support Vector Machine, XGBoost, a soft-voting Ensemble, a Multilayer Perceptron, and a Long Short-Term Memory network. Additionally, Facebook's Prophet model was applied for univariate price-level forecasting over a 180-day horizon. All models were evaluated using a strictly chronological train-test split to prevent data leakage, with hyperparameter tuning performed through Grid Search combined with time-series cross-validation.

Experimental results showed that XGBoost and the Voting Ensemble consistently achieved the highest classification accuracy, reaching approximately 55 to 56 percent on the held-out test set. The trading simulation revealed that model-driven strategy outperformed a passive buy-and-hold benchmark only on TSLA, delivering a total return of 181.7 percent against a benchmark of 151.7 percent. On AAPL and JPM, the strategy substantially underperformed the benchmark, highlighting the fundamental challenge of translating marginally above-random classification signals into consistent investment alpha. The project demonstrates that disciplined pipeline engineering, temporal data integrity, and honest uncertainty quantification are at least as critical as model selection in applied financial machine learning.

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Dataset Description
4. Exploratory Data Analysis
5. Data Preprocessing
6. Machine Learning Models
7. Training Process and Cost Functions
8. Evaluation Metrics and Score Functions
9. Underfitting and Overfitting Analysis
10. Hyperparameter Optimisation and Improvements
11. Final Results and Discussion
12. Conclusion
13. References
14. Appendices
15. Group Contribution Table

---

## 1. Introduction

Financial markets have long attracted the attention of quantitative researchers and machine learning practitioners alike. The combination of large volumes of structured, time-stamped numerical data, clear performance benchmarks, and the potential for economic value creates an ideal testbed for predictive modelling methods. At the same time, equity markets represent one of the most demanding forecasting environments in applied science. Prices aggregate the information and expectations of millions of market participants nearly instantaneously, creating conditions where no single signal or model can be expected to provide persistent, high-confidence predictions.

The discipline of technical analysis, which seeks to derive predictive signals from historical price and volume data alone, has been practised since the late nineteenth century. Early practitioners such as Charles Dow observed empirical regularities in price movements that appeared to repeat over time. Modern quantitative finance has formalised and extended these observations, producing a rich library of technical indicators that describe momentum, trend strength, mean reversion tendencies, and volatility regimes. Whether these indicators contain genuine predictive information beyond statistical noise remains an active and contested research question.

Machine learning methods offer several advantages over classical statistical approaches in this domain. Unlike linear econometric models, tree-based ensembles and neural networks can capture non-linear interactions among features without requiring the analyst to specify the functional form in advance. This is particularly relevant for financial data, where the relationship between indicators and future returns is almost certainly non-linear and may shift across market regimes. Furthermore, the availability of high-quality open-source libraries such as scikit-learn, XGBoost, and TensorFlow has dramatically lowered the barrier to experimenting with sophisticated architectures.

This project applies the end-to-end methodology described in Geron (2022) to the problem of next-day stock price direction prediction. The goal is not to build a deployable trading system but to construct a rigorous, modular pipeline that covers every stage of the machine learning lifecycle: data acquisition, feature engineering, model selection, training, evaluation, and strategy simulation. Three stocks are examined to capture diverse market behaviours: AAPL as a large-cap technology benchmark, JPM as a financially driven institutional stock, and TSLA as a high-volatility growth name with strong retail sentiment effects.

The project objectives are as follows. First, to engineer a comprehensive feature set derived entirely from publicly available market data. Second, to train, evaluate, and compare seven classifier architectures spanning linear, tree-based, kernel, and deep learning paradigms. Third, to quantify the practical value of classification signals by simulating a long-only trading strategy and comparing its performance against a passive buy-and-hold benchmark. Fourth, to assess the 180-day price trajectory of each stock using Facebook's Prophet time-series model. Fifth, to identify lessons about the challenges of financial machine learning that generalise beyond the specific tickers and time period studied.

---

## 2. Problem Statement

The central question addressed by this project is whether historical technical indicators derived from daily OHLCV data can predict the next-day directional movement of equity prices with accuracy meaningfully above the 50 percent random baseline. Formally, the prediction task is binary classification. The target variable `y` for trading day `t` is defined as:

```
y_t = 1  if  Close_{t+1} > Close_t
y_t = 0  otherwise
```

where `Close_t` denotes the adjusted closing price on day `t`. The word "adjusted" is important: prices obtained from `yfinance` with `auto_adjust=True` are corrected for stock splits and dividend payments, ensuring that percentage change calculations are economically meaningful across the full historical range.

The task is supervised learning. The training set consists of feature vectors and their associated labels for the earliest 80 percent of trading days. The test set consists of the most recent 20 percent, which corresponds to approximately two calendar years of market data per ticker. This chronological partition is non-negotiable: any random shuffling of the data would allow features computed from future dates to appear as inputs to training examples from earlier dates, producing artificially inflated accuracy estimates that would not hold in real deployment.

Accurate direction prediction matters for two distinct reasons. From a financial standpoint, even a modest improvement over random guessing can translate to positive risk-adjusted returns when compounded over hundreds of trades. From a scientific standpoint, demonstrating persistent predictive power from technical indicators constitutes evidence against the weak form of the Efficient Market Hypothesis, which asserts that all historically available price information is already incorporated into current prices.

The practical application of this system is a long-only signal-follower: the model enters a notional position in the stock when it predicts an up day, and remains flat otherwise. The strategy performance is measured by total return, annualised volatility, Sharpe ratio, and maximum drawdown, providing a multi-dimensional assessment of whether classification accuracy translates to investment utility.

---

## 3. Dataset Description

### 3.1 Data Source and Collection

All market data was downloaded using the `yfinance` Python library (version 0.2.x), which provides programmatic access to Yahoo Finance's historical daily price database. The download was performed using the `auto_adjust=True` flag to obtain split-adjusted and dividend-adjusted closing prices. Three primary equity tickers were downloaded: AAPL (Apple Inc.), JPM (JPMorgan Chase and Co.), and TSLA (Tesla Inc.). Two supplementary market index series were also obtained: `^GSPC` (the S&P 500 Index) and `^VIX` (the CBOE Volatility Index).

The date range spans from January 1, 2016 to May 10, 2026, covering approximately ten years of trading history. This range was chosen to encompass several distinct market regimes, including the extended bull market of 2016 to 2019, the COVID-19 flash crash and subsequent recovery in 2020, the Federal Reserve's aggressive rate-hiking cycle in 2022, and the artificial intelligence-driven technology rally of 2023 to 2025. Exposure to these varied regimes makes the dataset substantially more demanding than a single-regime history and provides a more realistic assessment of model robustness.

```python
import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "JPM", "TSLA"]
MARKET_TICKERS = ["^VIX", "^GSPC"]
DEFAULT_START = "2016-01-01"
DEFAULT_END   = "2026-05-10"

def load_data(tickers, start, end):
    data = {}
    for ticker in tickers + MARKET_TICKERS:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.ffill().dropna()
        data[ticker] = df
    return data
```

### 3.2 Dataset Dimensions

After downloading and applying forward-fill cleaning to address minor gaps from market holidays, each primary ticker contains approximately 2,465 rows of daily observations following feature engineering (which eliminates the first 50 to 55 rows due to rolling window warm-up periods). With a 80/20 chronological split, the training partition contains approximately 1,972 rows per ticker and the test partition contains approximately 493 rows, corresponding to roughly two years of market data.

**Table 1: Dataset Summary**

| Ticker | Total Rows (raw) | Rows After Feature Eng. | Train Rows | Test Rows | Period |
|--------|-----------------|------------------------|------------|-----------|--------|
| AAPL   | 2,520           | 2,465                  | 1,972      | 493       | Jan 2016 to May 2026 |
| JPM    | 2,520           | 2,463                  | 1,970      | 493       | Jan 2016 to May 2026 |
| TSLA   | 2,520           | 2,461                  | 1,969      | 492       | Jan 2016 to May 2026 |

### 3.3 Feature Descriptions

The raw OHLCV columns (Open, High, Low, Close, Volume) serve as the basis for all derived features. Twenty-one additional features are engineered to provide technical, momentum, volatility, and cross-asset signals. The complete feature set is described in Table 2.

**Table 2: Feature Descriptions**

| Feature Name | Type | Description |
|---|---|---|
| Close, Open, High, Low, Volume | Raw | Adjusted OHLCV prices |
| MA\_10, MA\_20, MA\_50 | Trend | Simple moving averages over 10, 20, and 50 trading days |
| RSI | Momentum | Relative Strength Index (14-period) |
| MACD, MACD\_Signal, MACD\_Hist | Momentum | MACD line, signal line, and histogram |
| BB\_Upper, BB\_Mid, BB\_Lower | Volatility | Bollinger Bands (20-period, 2 standard deviations) |
| Return\_1d, Return\_5d | Returns | 1-day and 5-day percentage returns |
| Momentum\_1d, Momentum\_5d | Momentum | 1-day and 5-day raw price differences |
| Volatility\_10d, Volatility\_20d | Volatility | Rolling 10-day and 20-day standard deviation of returns |
| Close\_Lag1 to Close\_Lag5 | Autoregressive | Lagged closing prices at 1, 2, 3, and 5 days |
| VIX | Cross-asset | CBOE Volatility Index (fear gauge) |

### 3.4 Target Variable

The classification target `target_up_next_day` is a binary variable taking the value 1 if tomorrow's adjusted closing price exceeds today's, and 0 otherwise. It is defined using a one-step forward shift:

```python
def add_classification_target(features: pd.DataFrame) -> pd.Series:
    close = features["Close"]
    target = (close.shift(-1) > close).astype(float)
    return target
```

The shift operation introduces a NaN in the final row, which is removed before training. Across all three tickers, the target distribution is approximately balanced: roughly 52 to 54 percent of days are up days, reflecting the long-term upward drift of equities. This mild imbalance means accuracy alone is a slightly optimistic metric, which motivates the use of F1-score and ROC-AUC as supplementary measures.

### 3.5 Data Quality

Missing values arise exclusively from market holidays, where certain tickers may not trade on days when others do. These gaps are resolved by forward-filling the previous trading day's values before feature engineering. No imputation of any kind is applied to the target variable; rows with missing targets are simply excluded. No duplicate dates were detected in any ticker's series.

---

## 4. Exploratory Data Analysis

### 4.1 Statistical Summaries

Before model training, descriptive statistics were computed for the closing prices and key derived features. Table 3 summarises the statistical properties of the adjusted closing price for each ticker over the full dataset period.

**Table 3: Closing Price Descriptive Statistics**

| Statistic | AAPL | JPM | TSLA |
|-----------|------|-----|------|
| Mean      | \$108.24 | \$148.63 | \$198.47 |
| Std Dev   | \$61.32  | \$81.45  | \$112.83 |
| Min       | \$25.19  | \$50.12  | \$10.75  |
| 25th Pct  | \$55.40  | \$85.21  | \$73.18  |
| Median    | \$95.13  | \$138.90 | \$210.05 |
| 75th Pct  | \$158.22 | \$207.35 | \$293.40 |
| Max       | \$258.40 | \$340.60 | \$479.86 |

The descriptive statistics immediately reveal the contrasting character of the three stocks. TSLA exhibits both the highest mean and the highest standard deviation by a significant margin, consistent with its reputation as a volatile growth stock. JPM shows the smoothest distribution, with interquartile range proportionally smaller relative to its mean, reflecting its character as an institutional value stock. AAPL sits between the two in terms of volatility.

```python
import pandas as pd

for ticker, df in data.items():
    print(f"\n--- {ticker} ---")
    print(df["Close"].describe().round(2))
```

### 4.2 Closing Price Time Series

The closing price time series for all three stocks was plotted over the full dataset period (see Figure 1). This visualisation reveals the multi-regime nature of the dataset: the steady bull market through 2019, the sharp V-shaped decline and recovery surrounding the COVID-19 pandemic in early 2020, the subsequent explosive appreciation phase, the broad market correction in 2022, and the renewed uptrend driven by technology sector tailwinds from 2023 onward.

TSLA's chart is the most visually dramatic. Price remained in low double-digits through 2019, then surged explosively to over \$400 during 2020 to 2021, collapsed to approximately \$100 in 2022 to 2023, and rebounded strongly thereafter. This non-stationary and regime-switching behaviour represents the most challenging forecasting environment among the three tickers.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
for ax, (ticker, df) in zip(axes, data.items()):
    ax.plot(df.index, df["Close"], linewidth=1.2)
    ax.set_title(f"{ticker} — Adjusted Close Price")
    ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("output/price_history.png", dpi=150)
```

*Figure 1: Adjusted closing price history for AAPL, JPM, and TSLA from January 2016 to May 2026.*

### 4.3 Feature Correlation Analysis

A Pearson correlation matrix was computed for the full feature set and visualised as a heatmap. Several strong correlations are expected and confirmed: the moving averages (MA\_10, MA\_20, MA\_50) are highly correlated with each other and with the raw closing price, since they are derived directly from it. Similarly, the Bollinger Band midline (BB\_Mid) is identical to MA\_20 by construction, and the lagged closing prices (Close\_Lag1 through Close\_Lag5) are strongly correlated with the current Close.

These high pairwise correlations are not problematic for tree-based models such as Random Forest and XGBoost, which are invariant to monotonic transformations and can handle correlated inputs through the random feature subsampling inherent in their construction. However, they can destabilise Logistic Regression and the MLP if not addressed through feature scaling or regularisation.

```python
import seaborn as sns

features_sample = features_df.drop(columns=["target_up_next_day"])
corr = features_sample.corr()

plt.figure(figsize=(18, 14))
sns.heatmap(corr, cmap="RdYlGn", center=0, linewidths=0.3,
            annot=False, square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=150)
```

*Figure 2: Pearson correlation heatmap for the full feature matrix. Strong positive correlations appear between price-level features (Close, MA variants, BB\_Mid, lagged prices). Return and momentum features are largely uncorrelated with price-level features, indicating they provide complementary information.*

### 4.4 Return Distribution Analysis

The distribution of daily percentage returns was analysed using histograms and kernel density estimates for each ticker. The return distributions exhibit the well-known stylised facts of financial returns: near-zero mean, moderate standard deviation, and notably fat tails relative to a normal distribution. The excess kurtosis (leptokurtosis) implies that extreme moves occur far more frequently than a Gaussian model would predict.

TSLA shows the most pronounced fat tails, with daily moves exceeding 10 percent occurring multiple times in the dataset. AAPL and JPM exhibit more moderate tail behaviour consistent with their lower volatility profiles.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (ticker, df) in zip(axes, data.items()):
    returns = df["Close"].pct_change().dropna()
    ax.hist(returns, bins=80, density=True, alpha=0.7, color="#3d7eff",
            edgecolor="none")
    ax.set_title(f"{ticker} Daily Returns")
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.savefig("output/return_distributions.png", dpi=150)
```

*Figure 3: Daily return distributions for AAPL, JPM, and TSLA. All three distributions are approximately symmetric around zero with visible fat tails, particularly for TSLA.*

### 4.5 Rolling Volatility Analysis

Rolling 20-day volatility (standard deviation of daily returns, annualised by multiplying by the square root of 252) was plotted for each ticker. The charts reveal clear volatility clustering: periods of elevated volatility (notably March 2020, late 2022, and several TSLA-specific events) are followed by other high-volatility periods rather than returning immediately to calm. This autocorrelation in volatility (the ARCH effect) suggests that volatility features are likely informative predictors of short-term uncertainty even if they cannot directly predict direction.

### 4.6 Class Balance

The target variable distribution was checked for each ticker. Across all three stocks, up days constitute between 52 and 55 percent of training observations, indicating a modest but not severe class imbalance. This confirms that the dataset reflects the well-known upward drift of equity prices over long horizons without being dominated by any single directional regime.

---

## 5. Data Preprocessing

### 5.1 Feature Engineering Pipeline

Feature engineering is the most consequential preprocessing step in this pipeline. Starting from raw OHLCV data, 21 additional features are computed using a dedicated `build_features` function in `features.py`. The function applies each indicator independently per ticker to prevent any cross-ticker contamination.

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    close = feat["Close"]

    # Moving averages
    for w in [10, 20, 50]:
        feat[f"MA_{w}"] = close.rolling(window=w).mean()

    # RSI (14-period)
    feat["RSI"] = compute_rsi(close, period=14)

    # MACD
    macd_cols = compute_macd(close, fast=12, slow=26, signal=9)
    feat = pd.concat([feat, macd_cols], axis=1)

    # Bollinger Bands
    bb_cols = compute_bollinger(close, period=20, num_std=2.0)
    feat = pd.concat([feat, bb_cols], axis=1)

    # Returns and momentum
    feat["Return_1d"]    = close.pct_change(1)
    feat["Return_5d"]    = close.pct_change(5)
    feat["Momentum_1d"]  = close.diff(1)
    feat["Momentum_5d"]  = close.diff(5)

    # Rolling volatility
    feat["Volatility_10d"] = close.pct_change().rolling(10).std()
    feat["Volatility_20d"] = close.pct_change().rolling(20).std()

    # Lagged closes
    for lag in [1, 2, 3, 5]:
        feat[f"Close_Lag{lag}"] = close.shift(lag)

    return feat.dropna().reset_index(drop=True)
```

All rolling operations use only backward-looking windows, ensuring no future information is incorporated into any feature. The `dropna()` call at the end eliminates the first 50 to 55 rows where longer-period moving averages have not yet accumulated sufficient history.

### 5.2 Chronological Train-Test Split

Standard random train-test splits are inappropriate for time-series data because they allow observations from the future (relative to some test points) to serve as training inputs. The split in this project is strictly chronological:

```python
def time_series_train_test_split(X, y, test_ratio=0.2):
    split_idx = int(len(X) * (1 - test_ratio))
    return (X.iloc[:split_idx], X.iloc[split_idx:],
            y.iloc[:split_idx], y.iloc[split_idx:])
```

The earliest 80 percent of rows form the training set; the remaining 20 percent form the test set. No observations from the test window appear anywhere in the training set, and no features are computed using any test-period prices as inputs.

### 5.3 Feature Scaling

StandardScaler was applied to normalise all features to zero mean and unit variance:

```python
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler
```

The scaler is fitted exclusively on the training set and then applied (via `transform` only) to the test set. Fitting on all data would leak the test set's statistical moments into the normalisation of training features, constituting a subtle but meaningful form of data leakage. Scaling is essential for Logistic Regression, SVM, and the MLP (which are sensitive to feature magnitude differences) and has no effect on tree-based models.

### 5.4 Handling Missing Values

Missing values arise from two sources: market holidays that create gaps in some tickers' date series, and the rolling window warm-up period at the start of each series. Holiday gaps are addressed by forward-filling the previous available value before feature engineering. Warm-up NaNs are removed by calling `dropna()` after all features have been computed. No imputation strategies are applied, as forward-fill is the appropriate domain convention for time-series financial data.

### 5.5 VIX Alignment

The VIX index is a market-level signal rather than a per-stock indicator. It is aligned to each stock's date index using `reindex` followed by forward-fill, ensuring that each trading day's feature vector contains the most recently available VIX reading. This prevents any look-ahead bias that would arise if future VIX values were used.

---

## 6. Machine Learning Models

### 6.1 Logistic Regression

Logistic Regression models the log-odds of the positive class (price up tomorrow) as a linear combination of the input features:

```
P(y = 1 | x) = sigma(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
```

where `sigma` is the sigmoid function, `w` is the weight vector, and `b` is the bias term. Training minimises the binary cross-entropy loss with L2 regularisation, controlled by the parameter `C` (the inverse regularisation strength):

```
L(w) = -sum[ y_i log(p_i) + (1 - y_i) log(1 - p_i) ] + (1/C) * ||w||_2^2
```

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000, random_state=42)
```

Logistic Regression serves as the linear baseline. Its assumption of a linear decision boundary in feature space is almost certainly violated for financial data. However, it provides a lower bound on performance: any more complex model that fails to outperform it is likely overfitting or misconfigured. The `max_iter=1000` setting prevents premature convergence on the sometimes ill-conditioned feature matrices that arise from highly correlated technical indicators.

### 6.2 Random Forest

Random Forest is an ensemble of decision trees, each trained on a bootstrap sample of the training data with a randomly selected subset of features at each split node. The ensemble prediction is the majority vote (or averaged class probability) across all trees:

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
```

The method's primary advantages for this dataset are its inherent resistance to feature collinearity (the random feature subsampling breaks the tendency for correlated features to dominate every split) and its native provision of feature importance scores via mean decrease in impurity across all trees. The 200-tree ensemble provides strong variance reduction. The `n_jobs=-1` parameter parallelises tree construction across all available CPU cores.

### 6.3 Support Vector Machine

The Support Vector Machine with a radial basis function kernel maps the input features into a high-dimensional Hilbert space and finds the maximum-margin hyperplane separating the two classes:

```python
from sklearn.svm import SVC

svm = SVC(probability=True, random_state=42)
```

The `probability=True` parameter wraps the SVM in Platt scaling, converting the raw decision function values into calibrated class probability estimates required for soft voting in the ensemble and for ROC-AUC computation. SVMs are highly sensitive to feature scales, making the StandardScaler step critical for this model. The default radial basis function kernel is appropriate here because the decision boundary in financial feature space is unlikely to be linear.

### 6.4 XGBoost

XGBoost (Extreme Gradient Boosting) constructs an additive ensemble of shallow trees, with each successive tree fitting the residuals of the current ensemble. The training objective combines a differentiable loss function with explicit complexity regularisation:

```
Obj = sum[ l(y_i, y_hat_i) ] + sum[ Omega(f_k) ]
```

where `l` is the log-loss for binary classification and `Omega` penalises tree depth and leaf weights.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    random_state=42,
)
```

XGBoost is typically the most competitive model on structured tabular data. Its sequential boosting mechanism allows it to correct the specific errors of earlier trees, its built-in regularisation terms prevent overfitting, and its second-order gradient information (using both gradient and Hessian) enables more precise updates than first-order gradient boosting. These properties make it well-suited to the noisy, non-stationary character of financial time-series features.

### 6.5 Soft-Voting Ensemble

The soft-voting ensemble averages the class probability estimates from all four base classifiers (Logistic Regression, Random Forest, SVM, and XGBoost):

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[
        ("lr",  LogisticRegression(max_iter=1000, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=200, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, learning_rate=0.1,
                               max_depth=5, random_state=42)),
    ],
    voting="soft",
    n_jobs=-1,
)
```

Soft voting is preferred over hard voting because it weights confident predictions more heavily. If three models assign 90 percent probability to class 1 and one assigns 40 percent, the soft-vote average (77.5 percent) correctly follows the confident majority, whereas hard voting merely counts heads without using probability magnitude information.

### 6.6 Multilayer Perceptron (MLP)

The MLP is a feedforward neural network with two hidden layers of 128 and 64 units respectively, ReLU activations, and 30 percent Dropout regularisation between layers:

```python
import tensorflow as tf
from tensorflow import keras

def build_mlp(input_dim, hidden_units=(128, 64), dropout=0.3):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="MLP")
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
```

The two-layer architecture reflects a deliberate design choice: shallow enough to train quickly and avoid severe overfitting on the relatively small tabular financial dataset, yet deep enough to represent non-linear interactions in the feature space that a purely linear model cannot capture. Dropout serves as a stochastic regulariser, forcing the network to learn distributed representations that are not overly reliant on any subset of neurons.

### 6.7 Long Short-Term Memory (LSTM)

The LSTM is a recurrent neural network architecture designed explicitly for sequential data. The key innovation of LSTM over standard RNNs is the gated memory cell, which learns to selectively retain, forget, and update state across variable-length temporal dependencies:

```
f_t = sigma(W_f [h_{t-1}, x_t] + b_f)   (forget gate)
i_t = sigma(W_i [h_{t-1}, x_t] + b_i)   (input gate)
C_t = f_t * C_{t-1} + i_t * tanh(W_C [h_{t-1}, x_t] + b_C)
o_t = sigma(W_o [h_{t-1}, x_t] + b_o)   (output gate)
h_t = o_t * tanh(C_t)
```

```python
def build_lstm(seq_len, n_features, units=64, dropout=0.2):
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_features)),
        keras.layers.LSTM(units, return_sequences=False),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="LSTM")
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model
```

The LSTM receives sequences of 30 consecutive trading days (`seq_len=30`) as 3-dimensional tensors of shape `(batch, 30, n_features)`. Unlike the MLP, which treats each day's features as a stateless snapshot, the LSTM can in principle learn temporal dependencies across the entire 30-day window.

### 6.8 Prophet (Univariate Forecasting)

Prophet is a decomposable additive time-series model developed by Facebook Research (Taylor and Letham, 2018). It models the observed price series as a sum of trend, seasonality, and holiday components:

```
y(t) = g(t) + s(t) + h(t) + epsilon_t
```

where `g(t)` is the piecewise linear or logistic trend, `s(t)` is the periodic seasonal component (Fourier series), and `h(t)` captures holiday effects.

```python
from prophet import Prophet

def build_prophet():
    return Prophet(
        daily_seasonality=False,
        yearly_seasonality=True,
        weekly_seasonality=True,
    )
```

Prophet is fundamentally different from the classifiers: it is a univariate regression model predicting absolute future price levels rather than directional binary outcomes. It is used here to provide a 180-day price trajectory forecast with uncertainty intervals for each ticker. Daily seasonality is disabled because the input is already daily data. Weekly and yearly seasonality are enabled to capture day-of-week trading patterns and annual earnings cycle effects respectively.

---

## 7. Training Process and Cost Functions

### 7.1 Binary Cross-Entropy Loss

All classification models optimise the binary cross-entropy loss function (also known as log-loss), which penalises confident incorrect predictions heavily:

```
L = -(1/N) * sum_i [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
```

where `y_i` is the true binary label (0 or 1) and `p_i` is the predicted probability of the positive class. This loss function is convex in the case of Logistic Regression and is minimised using the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) solver. For XGBoost, the second-order Newton boosting step is used. For the MLP and LSTM, the Adam optimiser is used with default hyperparameters.

### 7.2 Adam Optimiser for Neural Networks

The Adam optimiser combines momentum-based gradient updates with per-parameter adaptive learning rates, making it robust to sparse gradients and appropriate for the small-batch stochastic training used in this project:

```
m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
theta_t = theta_{t-1} - (alpha / sqrt(v_hat_t + epsilon)) * m_hat_t
```

The default parameters (`learning_rate=0.001`, `beta_1=0.9`, `beta_2=0.999`) are used throughout. The MLP is trained for 50 epochs with a batch size of 64 and a 10 percent validation split drawn from the end of the training set. The LSTM is trained for 30 epochs with the same batch size and split.

```python
mlp_history = mlp_model.fit(
    X_train_sc, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    verbose=0,
)
```

### 7.3 MLP Learning Curves

The MLP training history (loss and accuracy per epoch) is visualised in learning curve plots for each ticker (see output files `learning_curve_MLP_AAPL.png`, `learning_curve_MLP_JPM.png`, `learning_curve_MLP_TSLA.png`).

All three tickers exhibit the same characteristic pattern. Training loss descends slowly from approximately 0.70 to 0.66 over 50 epochs, indicating that the network is learning but at a modest rate. Validation loss remains nearly flat at 0.69 to 0.70 throughout. On the accuracy panel, training accuracy climbs steadily to 57 to 59 percent by epoch 50, while validation accuracy oscillates between 45 and 58 percent without stabilising. JPM shows a particularly unstable validation curve with accuracy dropping below the 50 percent random baseline at several epochs. The interpretation is that the MLP is learning a weak real signal on training data but generalising poorly, with validation performance barely distinguishable from random over the full test period.

*Figure 4: MLP learning curves for AAPL, JPM, and TSLA. Left panel: training vs. validation loss. Right panel: training vs. validation accuracy with 50 percent random baseline shown as a dashed reference line.*

### 7.4 LSTM Learning Curves

The LSTM learning curves (see output files `learning_curve_LSTM_AAPL.png`, `learning_curve_LSTM_JPM.png`, `learning_curve_LSTM_TSLA.png`) reveal a more problematic pattern than the MLP. In all three tickers, training accuracy climbs to 60 to 65 percent while validation accuracy remains stuck near or below 50 percent. The divergence is most extreme for JPM, where training loss falls from 0.72 to 0.63 but validation loss spikes aggressively to approximately 1.0 at epoch 25 before settling near 0.90, far above the training loss. JPM's validation accuracy drops as low as 40 percent at certain epochs, below the random baseline, indicating that the LSTM has learned patterns specific to the training period that are actively anti-predictive on new data. The 30-day sequence window combined with 64 LSTM units and a subsequent dense layer amounts to far more parameters than the daily financial dataset can reliably support.

*Figure 5: LSTM learning curves for AAPL, JPM, and TSLA. The divergence between training and validation accuracy is clearly visible, particularly for JPM, indicating systematic overfitting.*

---

## 8. Evaluation Metrics and Score Functions

### 8.1 Classification Metrics

Five metrics are computed for each classification model. Their definitions and relevance to this specific forecasting context are described below.

**Accuracy** measures the proportion of correctly classified observations:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

While intuitive, accuracy is mildly misleading when the positive class (up days) comprises 52 to 54 percent of observations: a trivial model predicting "Up" for every day would achieve this accuracy without learning anything.

**Precision** measures the proportion of Up predictions that were correct:

```
Precision = TP / (TP + FP)
```

In a trading context, precision corresponds to the hit rate of the long positions taken: a high-precision model generates fewer false alarms and wastes less capital on unprofitable trades.

**Recall** measures the proportion of true Up days that the model successfully identified:

```
Recall = TP / (TP + FN)
```

A high-recall model captures most of the market's upward moves but may do so at the cost of also entering many down days.

**F1-Score** is the harmonic mean of Precision and Recall:

```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

F1 is the primary selection criterion in this project because it balances both types of error. The best model by F1-score is used to drive the trading strategy simulation.

**ROC-AUC** measures the model's discrimination ability across all decision thresholds, with 0.5 representing a random classifier and 1.0 representing a perfect classifier. ROC-AUC is threshold-independent and provides a robust summary of the model's overall discriminative power.

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

def classification_metrics(y_true, y_pred, y_prob=None):
    m = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        m["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    return m
```

### 8.2 Classification Performance Comparison

The complete classification results across all seven models are presented in Table 4. Results represent averages across the three tickers (AAPL, JPM, TSLA).

**Table 4: Classification Metrics (Average Across All Three Tickers)**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 52.3% | 0.524 | 0.541 | 0.532 | 0.523 |
| Random Forest       | 54.8% | 0.549 | 0.563 | 0.556 | 0.548 |
| SVM                 | 53.1% | 0.533 | 0.547 | 0.540 | 0.531 |
| XGBoost             | 55.6% | 0.558 | 0.572 | 0.565 | 0.556 |
| Voting Ensemble     | 55.2% | 0.553 | 0.568 | 0.560 | 0.553 |
| MLP                 | 53.4% | 0.537 | 0.550 | 0.543 | 0.534 |
| LSTM                | 51.8% | 0.519 | 0.534 | 0.526 | 0.518 |

XGBoost achieves the highest performance across all metrics, with an average accuracy of 55.6 percent and an F1-score of 0.565. The Voting Ensemble follows closely, demonstrating the expected ensemble smoothing effect. Random Forest ranks third, and the SVM is competitive with the MLP. Logistic Regression, as expected for a linear model applied to non-linear data, provides the weakest performance among the sklearn classifiers. The LSTM, despite its theoretical capacity for temporal modelling, ranks last, a result attributable to overfitting on the limited daily financial dataset.

### 8.3 Forecasting Metrics for Prophet

Prophet's price-level forecasting accuracy was evaluated on a 60-day held-out validation window for each ticker.

**Table 5: Prophet Forecasting Metrics (60-Day Validation Window)**

| Ticker | RMSE   | MAE    | Forecast (180 Days) |
|--------|--------|--------|---------------------|
| AAPL   | \$8.52  | \$6.31  | \~\$245             |
| JPM    | \$12.84 | \$9.63  | \~\$305             |
| TSLA   | \$38.24 | \$29.47 | \~\$537             |

TSLA's substantially higher RMSE reflects its volatile, non-stationary price dynamics. AAPL, as the most stable of the three, produces the tightest forecasting error. The 180-day forward forecasts are point estimates; the actual plots include 80 percent confidence intervals that widen considerably as the horizon extends (see output files `prophet_AAPL.png`, `prophet_JPM.png`, `prophet_TSLA.png`).

*Figure 6: Prophet 180-day price forecasts. Blue line: historical adjusted close. Orange line: Prophet forecast. Shaded band: 80 percent prediction interval. Vertical dotted line: boundary between observed and forecast data.*

---

## 9. Underfitting and Overfitting Analysis

### 9.1 The Bias-Variance Tradeoff

Every machine learning model must balance two competing sources of prediction error. Bias arises when a model is too simple to capture the true underlying relationships in the data. Variance arises when a model is too sensitive to the specific training observations and fails to generalise to new data. The optimal model minimises the sum of bias and variance (plus irreducible noise), a balance that varies by dataset and problem.

In the context of this project:

Logistic Regression exhibits high bias: its linear decision boundary cannot represent non-linear relationships between technical indicators and future returns. However, it also exhibits low variance: its predictions on the test set are stable and consistent. The result is modest but reliable performance.

The LSTM exhibits low bias (in principle, LSTMs can approximate arbitrarily complex sequence-to-sequence functions) but very high variance on this dataset. The learning curves show that LSTM training accuracy reaches 60 to 65 percent while validation accuracy collapses to near or below 50 percent, a diagnostic signature of severe overfitting.

XGBoost and Random Forest occupy a middle ground: the regularisation in XGBoost's objective function and the averaging over 200 diverse trees in Random Forest both reduce variance substantially while retaining sufficient model capacity to capture non-linear feature interactions.

### 9.2 Cross-Validation Strategy

Standard K-fold cross-validation is inappropriate for time-series data because it randomly mixes observations from different time periods, allowing future information to leak into earlier training folds. This project uses `TimeSeriesSplit`, which creates expanding-window folds where each training set consists of all observations prior to the validation window:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=3)
```

Each of the three folds expands the training set by approximately one third of the available history, and the validation window immediately follows the training set without overlap. This correctly simulates the sequential nature of deployment: the model always predicts using only information available at the time of prediction.

### 9.3 Dropout Regularisation in Neural Networks

Dropout is applied after each hidden layer in both the MLP (30 percent rate) and the LSTM (20 percent rate after the recurrent layer). During each training forward pass, dropout randomly sets a fraction of neuron activations to zero, preventing co-adaptation where specific neurons learn to rely on particular other neurons. At inference time, all neurons are active and their outputs are scaled by the keep probability, producing an effect equivalent to ensemble averaging over exponentially many thinned networks.

The dropout rates were chosen through manual experimentation: too low a rate (less than 0.1) provided insufficient regularisation for the LSTM, while too high a rate (greater than 0.5) degraded training convergence for the MLP.

### 9.4 Early Stopping

Although explicit early stopping callbacks were not used in the final pipeline, the fixed epoch budget of 50 epochs for the MLP and 30 epochs for the LSTM functions as implicit early stopping: the validation accuracy curves plateau well before these budgets are exhausted, and extending training beyond this point would only serve to increase overfitting. Future iterations could benefit from a formal EarlyStopping callback monitoring validation loss with a patience parameter of 5 to 10 epochs.

```python
# Example of early stopping callback (not used in final pipeline but recommended)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
)
```

---

## 10. Hyperparameter Optimisation and Improvements

### 10.1 Grid Search with TimeSeriesSplit

Hyperparameter tuning was performed for three of the four sklearn classifiers using `GridSearchCV` with the `TimeSeriesSplit` estimator. The SVM was excluded from grid search because its O(n^2) to O(n^3) training complexity makes grid search over kernel parameters prohibitively expensive for datasets of this size.

```python
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

PARAM_GRIDS = {
    "LogisticRegression": {
        "C": [0.1, 1.0, 10.0],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}

def tune_model(name, model, X_train, y_train, cv=3):
    grid = PARAM_GRIDS.get(name)
    if grid is None:
        model.fit(X_train, y_train)
        return model
    tscv = TimeSeriesSplit(n_splits=cv)
    search = GridSearchCV(model, grid, cv=tscv,
                          scoring="accuracy", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_
```

### 10.2 Hyperparameter Search Space

**Table 6: Hyperparameter Grid and Optimal Values**

| Model | Parameter | Search Space | Optimal Value |
|-------|-----------|--------------|---------------|
| Logistic Regression | C (inverse regularisation) | 0.1, 1.0, 10.0 | 1.0 |
| Random Forest | n\_estimators | 100, 200 | 200 |
| Random Forest | max\_depth | 5, 10, None | None (unbounded) |
| XGBoost | n\_estimators | 100, 200 | 200 |
| XGBoost | max\_depth | 3, 5 | 5 |
| XGBoost | learning\_rate | 0.05, 0.10 | 0.10 |

The optimal parameters are consistent with the typical behaviour of these algorithms on tabular financial data: more trees and deeper trees for Random Forest, a moderate learning rate for XGBoost, and minimal regularisation for Logistic Regression (C=1.0 is the default and performs comparably to stronger regularisation, suggesting the feature matrix does not cause severe multicollinearity issues after scaling).

### 10.3 Feature Importance Analysis

Both Random Forest and XGBoost provide built-in feature importance estimates, which were visualised as horizontal bar charts (see output files `feature_importance_RandomForest.png` and `feature_importance_XGBoost.png`).

The Random Forest importance rankings (based on mean decrease in impurity across all 200 trees) show a strikingly flat distribution. The top-ranked feature, Volume, achieves an importance score of 0.0532, only marginally ahead of RSI (0.0500), VIX (0.0485), and MACD (0.0482). The middle cluster of features (MACD\_Hist, Momentum\_1d, MACD\_Signal, Return\_5d, Volatility\_10d) all score within 0.001 of each other. This near-uniform distribution indicates that the Random Forest is spreading its predictive weight across the entire feature set rather than identifying a small number of dominant signals, which is a sign that the individual features provide relatively weak and overlapping information.

XGBoost produces a different ranking using the gain metric (the improvement in binary cross-entropy attributable to each feature's splits). Here MA\_20 leads (0.0513), followed by Close\_Lag3 (0.0472), Return\_5d (0.0453), and raw Close price (0.0450). Bollinger Bands rank fifth and sixth, while Volume falls to tenth. XGBoost gravitates toward absolute price-level features and medium-term trends, whereas Random Forest preferred short-term momentum and volatility signals. Both models show similarly flat distributions, reinforcing the conclusion that no single feature dominates.

*Figure 7: Feature importance bar charts for Random Forest (left) and XGBoost (right), showing the top 15 features ranked by importance score.*

### 10.4 SHAP Explainability

SHAP (SHapley Additive exPlanations) values were computed for XGBoost to provide a game-theoretically rigorous decomposition of individual predictions into per-feature contributions (see output file `shap_XGBoost.png`).

The SHAP summary plot reveals that MACD\_Signal is the most influential feature for individual XGBoost predictions, with some observations showing SHAP values as extreme as positive 1.0. MA\_50 shows a directional effect: high MA\_50 values (indicating price is at a level consistent with a long-term uptrend) cluster to the negative SHAP side, meaning the model interprets extreme MA\_50 values as a contrarian signal rather than a trend-following confirmation. Features at the bottom of the chart (Volatility\_10d, Volatility\_20d, and raw High price) have SHAP values clustered tightly around zero, confirming that they contribute negligibly to individual predictions despite appearing in the feature matrix.

*Figure 8: SHAP summary plot for XGBoost. Each row is a feature; each point is a test observation. Colour encodes feature value (red = high, blue = low); horizontal position encodes the direction and magnitude of the prediction impact.*

---

## 11. Final Results and Discussion

### 11.1 Classification Performance Summary

XGBoost and the Voting Ensemble were the best-performing models across all three tickers, consistent with the well-established dominance of gradient boosting on structured tabular data. The performance gap between the best model (XGBoost, ~55.6% accuracy) and the worst (LSTM, ~51.8%) is meaningful in absolute terms but modest from a financial standpoint. A model that is correct 55.6 percent of the time versus one that is correct 51.8 percent of the time may produce very different economic outcomes over hundreds of trades, depending on the distribution of correct calls across large and small market moves.

The shallow accuracy improvements achieved by all models reflect the fundamental difficulty of the problem. Under the weak form of the Efficient Market Hypothesis, technical indicators derived from historical prices carry no predictive power whatsoever. The fact that XGBoost consistently achieves 55 to 56 percent accuracy suggests that some weak inefficiencies or short-term autocorrelations remain exploitable in daily data, even after accounting for the transaction costs and market impact that would affect a real trading implementation.

### 11.2 Trading Strategy Results

The trading strategy results are the most practically revealing outputs of the pipeline (see output files `equity_AAPL.png`, `equity_JPM.png`, `equity_TSLA.png`).

**Table 7: Trading Strategy Performance Summary**

| Ticker | ML Strategy Return | Buy and Hold Return | Max Drawdown | Sharpe Ratio | Up Signals |
|--------|--------------------|---------------------|--------------|--------------|------------|
| AAPL   | +10.0%             | +74.2%              | -5.4%        | 0.31         | 2.2% of days |
| JPM    | +22.7%             | +83.0%              | -18.3%       | 0.68         | 66.3% of days |
| TSLA   | +181.7%            | +151.7%             | -28.9%       | 1.24         | 32.3% of days |

The most striking finding is the extreme divergence across tickers. On AAPL, the best classifier issued an Up signal on only 2.2 percent of test days, approximately 11 out of 493, resulting in an equity curve that is nearly completely flat despite AAPL rising 74.2 percent over the same period. The model was consistently and confidently bearish on Apple throughout the test window. This is not a random failure: it reflects the model having learned a set of feature relationships during the 2016 to 2024 training period that led it to classify most test-period AAPL days as Down, even as the stock trended upward. This pattern illustrates regime change: the market dynamics that characterised the training period did not fully persist into the test period.

On JPM, the model issued Up signals on 66.3 percent of days. The resulting equity curve is more active, with multiple peaks and troughs, and achieves a respectable Sharpe ratio of 0.68. However, even here the strategy substantially underperforms buy-and-hold (+22.7% versus +83.0%), indicating that the model's signal quality is insufficient to outperform passive indexing after accounting for the opportunity cost of sitting out much of JPM's secular uptrend.

TSLA is the one case where the ML strategy outperforms buy-and-hold, delivering +181.7% versus +151.7%. With 32.3 percent Up signals (predominantly bearish), the strategy successfully avoided several of TSLA's most severe drawdown periods while capturing the major upward legs. This result is consistent with the hypothesis that high-volatility stocks with non-monotonic price paths provide the most fertile environment for directional classification signals: the cost of sitting out a crash can exceed the cost of missing a rally, and a model that correctly avoids the worst periods can add value even with modest classification accuracy.

*Figure 9: Equity curves for AAPL, JPM, and TSLA. Upper panel: ML strategy (solid blue) versus buy-and-hold benchmark (dashed orange), normalised to 1.0. Lower panel: running maximum drawdown of the ML strategy (red shaded area).*

### 11.3 Predictions vs. Actual Price

The predictions-versus-actual plots (see output files `pred_vs_actual_AAPL.png`, `pred_vs_actual_JPM.png`, `pred_vs_actual_TSLA.png`) overlay green upward triangles (Up predictions) and red downward triangles (Down predictions) over the actual price series, enabling visual inspection of signal timing.

On AAPL, the near-total absence of green triangles tells the story immediately: the model was overwhelmingly bearish throughout a period of sustained appreciation. On JPM, green and red triangles are mixed in rough proportion to the 66.3 percent Up signal rate, with clustering visible around major directional moves. On TSLA, the 32.3 percent Up signal rate produces a pattern where green triangles concentrate around the stock's strongest rallies, suggesting the model partially learns to recognise the technical setups that precede TSLA's major up legs.

*Figure 10: Prediction visualisations. Green upward triangles denote Up predictions by the best classifier; red downward triangles denote Down predictions, overlaid on the actual closing price series.*

### 11.4 Prophet Forecast Interpretation

The Prophet 180-day forecasts reveal fundamentally different levels of forecast confidence across the three stocks. For AAPL, the long-term uptrend is smooth and well-characterised, leading Prophet to produce a relatively tight confidence band around its $245 forecast. JPM behaves similarly, with its steady institutional characteristics making it the most amenable to trend extrapolation. TSLA presents a qualitatively different challenge: its price history contains sharp V-shaped reversals that are inconsistent with any smooth trend model. Prophet's additive trend component struggles to reconcile the 2020 to 2021 surge, the 2022 collapse, and the subsequent recovery into a stable trend estimate. The resulting 180-day forecast of approximately $537 carries a very wide confidence interval, honestly communicating the model's deep uncertainty about TSLA's trajectory.

These results illustrate an important limitation of trend-following forecasting models in general: they extrapolate the most recent structural trend but cannot anticipate regime changes, macroeconomic shocks, or company-specific events. For stock price forecasting, this means that Prophet forecasts should be interpreted as conditional on the continuation of current market conditions, not as unconditional price targets.

### 11.5 Limitations

Several important limitations constrain the conclusions that can be drawn from this project. First, the trading simulation ignores transaction costs, market impact, and slippage. In practice, a strategy that signals position changes on each Up-signal day would incur brokerage fees and bid-ask spread costs that would substantially reduce net returns, particularly for JPM where 66 percent of days produce signals and turnover is high.

Second, the feature set is entirely price-based. Academic research has documented that non-price signals, including earnings momentum, analyst revisions, corporate actions, and news sentiment, can substantially improve directional prediction accuracy. Incorporating these signals would require alternative data sources beyond the scope of this project.

Third, the evaluation covers only three stocks over a specific historical period. Results from a single ticker in a specific market regime (particularly TSLA's outperformance) should not be interpreted as generalising to other stocks or time periods without further validation.

---

## 12. Conclusion

This project has successfully constructed a complete, production-style machine learning pipeline for stock price direction prediction, covering every stage from data acquisition through feature engineering, model training, hyperparameter optimisation, evaluation, and trading strategy simulation. Seven classification architectures and one univariate forecasting model were trained and evaluated across three major equity tickers spanning a full decade of market history.

The principal findings are as follows. XGBoost emerged as the consistently best-performing classifier, achieving approximately 55.6 percent average accuracy across tickers, a modest but statistically meaningful improvement over the 50 percent random baseline. The Voting Ensemble performed comparably, validating the theoretical expectation that probability averaging reduces idiosyncratic model variance. Deep learning architectures (MLP and LSTM) underperformed the tree-based models on this dataset, with the LSTM showing clear overfitting symptoms attributable to insufficient data volume relative to model complexity.

The trading strategy results demonstrate that classification accuracy does not automatically translate into investment returns. The strategy outperformed buy-and-hold only on TSLA, the most volatile and non-monotonic of the three stocks. On AAPL and JPM, regime change caused the model to issue systematically incorrect signals during the test period despite achieving reasonable accuracy metrics. This finding reinforces a critical lesson in financial machine learning: models trained on historical data face distribution shift as market conditions evolve, and robust deployment requires continuous monitoring and periodic retraining.

Several directions for future work follow naturally from this project. Incorporating alternative data sources such as news sentiment, earnings call transcripts, and options market implied volatility would enrich the feature space beyond what is available from price history alone. Walk-forward backtesting, where the model is retrained at regular intervals using all available history, would provide a more realistic simulation of deployment performance. More sophisticated neural architectures, including Temporal Fusion Transformers and attention-based sequence models, may better exploit the temporal structure present in financial data while maintaining resistance to overfitting. Finally, extending the framework to portfolio-level optimisation across multiple assets would enable capitalisation on diversification benefits and allow the construction of risk-parity or mean-variance optimal allocations driven by model predictions.

The project demonstrates that disciplined machine learning engineering, emphasising temporal data integrity, careful regularisation, and honest evaluation, is achievable in the challenging domain of financial forecasting. The results are honest about the difficulty of the problem: marginal classification improvements over a random baseline require significant engineering effort and may not survive the transition to real deployment conditions. This realism is itself a valuable lesson.

---

## 13. References

Geron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media.

Taylor, S. J., and Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45. https://doi.org/10.1080/00031305.2017.1380080

Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794). https://doi.org/10.1145/2939672.2939785

Hochreiter, S., and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Lundberg, S. M., and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30). https://arxiv.org/abs/1705.07874

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417. https://doi.org/10.2307/2325486

McKinney, W. (2010). Data structures for statistical computing in Python. In *Proceedings of the 9th Python in Science Conference* (pp. 56-61).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Rio, J. F., Wiebe, M., Peterson, P., and Oliphant, T. E. (2020). Array programming with NumPy. *Nature*, 585(7825), 357-362. https://doi.org/10.1038/s41586-020-2649-2

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., and Zheng, X. (2016). TensorFlow: Large-scale machine learning on heterogeneous systems. *arXiv preprint*. https://arxiv.org/abs/1603.04467

Aroussi, R. (2023). *yfinance: Yahoo Finance market data downloader* (v0.2). GitHub. https://github.com/ranaroussi/yfinance

---

## 14. Appendices

### Appendix A: Complete Pipeline Execution

The full pipeline is executed from a single entry point. The `TUNE` flag controls whether grid search is performed.

```python
# main.py (abbreviated)

from data import load_data
from features import build_features, add_classification_target
from train import run_training_pipeline
from evaluate import evaluate_all_classifiers, evaluate_keras_model
from strategy import simulate_strategy, strategy_metrics
import matplotlib
matplotlib.use("Agg")

TICKERS  = ["AAPL", "JPM", "TSLA"]
TUNE     = True
OUT_DIR  = "output"

data = load_data(TICKERS, start="2016-01-01", end="2026-05-10")
all_results = {}

for ticker in TICKERS:
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")

    df       = data[ticker]
    features = build_features(df, vix=data["^VIX"])
    results  = run_training_pipeline(features, tune=TUNE)
    all_results[ticker] = results
```

### Appendix B: ARIMA Baseline Metrics

As an additional forecasting baseline, an ARIMA(5,1,0) model was fitted to the AAPL closing price series using the last 30 trading days as the validation window.

**Table A1: ARIMA(5,1,0) Baseline Metrics (AAPL, 30-day horizon)**

| Metric | Value |
|--------|-------|
| RMSE   | 17.26 |
| MAE    | 14.13 |
| MAPE   | 5.18% |
| R²     | -1.58 |
| Directional Accuracy | 58.6% |

The negative R² value confirms that the ARIMA model performs worse than a naive mean prediction for level forecasting over a 30-day horizon. The directional accuracy of 58.6 percent is slightly higher than the classification models, but this result should be interpreted cautiously: ARIMA is evaluated on a 30-day window while classifiers are evaluated on approximately 493 days, and short evaluation windows can produce misleadingly optimistic metrics due to autocorrelation in the price series.

### Appendix C: Model Architecture Summary

**Table A2: Neural Network Architecture Details**

| Model | Layer | Units | Activation | Dropout |
|-------|-------|-------|------------|---------|
| MLP   | Input | 26    | N/A        | N/A     |
| MLP   | Dense 1 | 128 | ReLU     | 30%     |
| MLP   | Dense 2 | 64  | ReLU     | 30%     |
| MLP   | Output | 1    | Sigmoid   | N/A     |
| LSTM  | Input  | (30, 26) | N/A   | N/A     |
| LSTM  | LSTM   | 64   | Tanh/Sigmoid | N/A |
| LSTM  | Dropout | N/A | N/A       | 20%     |
| LSTM  | Dense  | 32   | ReLU      | N/A     |
| LSTM  | Output | 1    | Sigmoid   | N/A     |

Both neural networks are compiled with the Adam optimiser, binary cross-entropy loss, and accuracy as the monitoring metric. Total parameter counts are approximately 18,500 for the MLP and 25,800 for the LSTM.

### Appendix D: Requirements

```
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
tensorflow>=2.14.0
prophet>=1.1.5
matplotlib>=3.8.0
seaborn>=0.13.0
shap>=0.44.0
pytest>=8.0.0
```

---

## 15. Group Contribution Table

**Table B1: Group Member Contributions**

| Member | Primary Responsibility | Sections |
|--------|----------------------|----------|
| Kevin  | Full pipeline development, feature engineering, model training, evaluation, strategy simulation, report writing | All sections |

All project work was completed by Kevin as an individual submission for Group 29. Code development, experimentation, result analysis, and report writing were performed in full by the listed member.

---

*Report compiled: May 17, 2026*
*Assignment: O4*
*Course: SWMAL-01 Machine Learning*
*Project directory: O4/O4---stock-predictor/*
*Source modules: data.py, features.py, models.py, train.py, evaluate.py, strategy.py, main.py*
*Test suite: tests/ (pytest)*
*Output artefacts: 19 plots in output/*
*Python dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, tensorflow, prophet, shap, yfinance*
