"""
models.py — Model definitions.

Classification models, deep learning (MLP / LSTM), Prophet forecasting,
and an ensemble combiner.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from prophet import Prophet

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Classification models
# ---------------------------------------------------------------------------

def get_logistic_regression() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=42)


def get_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, random_state=42,
                                  n_jobs=-1)


def get_svm() -> SVC:
    return SVC(probability=True, random_state=42)


def get_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )


def get_classifiers() -> dict:
    """Return a dict of name → classifier."""
    return {
        "LogisticRegression": get_logistic_regression(),
        "RandomForest": get_random_forest(),
        "SVM": get_svm(),
        "XGBoost": get_xgboost(),
    }


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def get_voting_ensemble(classifiers: dict | None = None) -> VotingClassifier:
    """Soft-voting ensemble over the provided classifiers."""
    if classifiers is None:
        classifiers = get_classifiers()
    estimators = [(name, clf) for name, clf in classifiers.items()]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


# ---------------------------------------------------------------------------
# Deep learning — MLP
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden_units: tuple[int, ...] = (128, 64),
              dropout: float = 0.3) -> keras.Model:
    """Simple feedforward neural network for binary classification."""
    model = keras.Sequential(name="MLP")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Deep learning — LSTM
# ---------------------------------------------------------------------------

def build_lstm(seq_len: int, n_features: int,
               units: int = 64, dropout: float = 0.2) -> keras.Model:
    """LSTM model for time-series classification."""
    model = keras.Sequential(name="LSTM")
    model.add(keras.layers.Input(shape=(seq_len, n_features)))
    model.add(keras.layers.LSTM(units, return_sequences=False))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def create_sequences(X: np.ndarray, y: np.ndarray,
                     seq_len: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Roll a fixed-length window over X to produce 3-D input for LSTM.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    seq_len : int

    Returns
    -------
    X_seq : np.ndarray of shape (n_samples - seq_len + 1, seq_len, n_features)
    y_seq : np.ndarray of shape (n_samples - seq_len + 1,)
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X) + 1):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i - 1])
    return np.array(Xs), np.array(ys)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def build_prophet() -> Prophet:
    """Create a Prophet model with default settings."""
    return Prophet(daily_seasonality=False, yearly_seasonality=True,
                   weekly_seasonality=True)


def fit_prophet(model: Prophet, dates, prices) -> Prophet:
    """Fit Prophet on date-price series.

    Parameters
    ----------
    model : Prophet
    dates : array-like of datetime
    prices : array-like of float

    Returns
    -------
    Prophet (fitted)
    """
    import pandas as pd
    prophet_df = pd.DataFrame({"ds": dates, "y": prices})
    model.fit(prophet_df)
    return model


def predict_prophet(model: Prophet, periods: int = 30,
                    freq: str = "B") -> "pd.DataFrame":
    """Generate future forecast from a fitted Prophet model."""
    import pandas as pd
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
