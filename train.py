"""
train.py — Training pipeline.

Handles time-series splitting, hyperparameter tuning, and model training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from features import build_features, add_classification_target, scale_features
from models import (
    get_classifiers,
    get_voting_ensemble,
    build_mlp,
    build_lstm,
    create_sequences,
)


# ---------------------------------------------------------------------------
# Time-series split helper
# ---------------------------------------------------------------------------

def time_series_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data chronologically (no shuffling)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

# Compact grids — extend if more compute is available
PARAM_GRIDS: dict[str, dict] = {
    "LogisticRegression": {"C": [0.1, 1.0, 10.0]},
    "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    },
}


def tune_model(name: str, model: object, X_train: np.ndarray,
               y_train: np.ndarray, cv: int = 3) -> object:
    """Run GridSearchCV if a param grid is available for *name*."""
    grid = PARAM_GRIDS.get(name)
    if grid is None:
        model.fit(X_train, y_train)
        return model

    tscv = TimeSeriesSplit(n_splits=cv)
    search = GridSearchCV(
        model, grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  {name} best params: {search.best_params_}  "
          f"CV accuracy: {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Train all sklearn classifiers
# ---------------------------------------------------------------------------

def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
) -> dict:
    """Train each classifier (with optional tuning) and return fitted models."""
    classifiers = get_classifiers()
    trained: dict = {}
    for name, clf in classifiers.items():
        print(f"  Training {name} …")
        if tune:
            clf = tune_model(name, clf, X_train, y_train)
        else:
            clf.fit(X_train, y_train)
        trained[name] = clf
    return trained


# ---------------------------------------------------------------------------
# Train ensemble
# ---------------------------------------------------------------------------

def train_ensemble(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """Fit a soft-voting ensemble of default classifiers."""
    ens = get_voting_ensemble()
    print("  Training VotingEnsemble …")
    ens.fit(X_train, y_train)
    return ens


# ---------------------------------------------------------------------------
# Train MLP
# ---------------------------------------------------------------------------

def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 50, batch_size: int = 64,
              validation_split: float = 0.1) -> tuple:
    """Build and train a feedforward MLP. Returns (model, history)."""
    model = build_mlp(input_dim=X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, verbose=0)
    return model, history


# ---------------------------------------------------------------------------
# Train LSTM
# ---------------------------------------------------------------------------

def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               seq_len: int = 30, epochs: int = 30,
               batch_size: int = 64) -> tuple:
    """Build and train an LSTM model. Returns (model, seq_len, history)."""
    X_seq, y_seq = create_sequences(X_train, y_train, seq_len=seq_len)
    model = build_lstm(seq_len=seq_len, n_features=X_seq.shape[2])
    history = model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=0)
    return model, seq_len, history


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(
    features: pd.DataFrame,
    tune: bool = True,
    lstm_epochs: int = 30,
    mlp_epochs: int = 50,
) -> dict:
    """End-to-end: split → scale → train all models.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix (output of ``build_features``).
    tune : bool
        Whether to perform GridSearchCV for sklearn models.
    lstm_epochs, mlp_epochs : int
        Number of training epochs.

    Returns
    -------
    dict with keys:
        "classifiers", "ensemble", "mlp", "lstm", "lstm_seq_len",
        "scaler", "X_train", "X_test", "y_train", "y_test",
        "X_train_raw", "X_test_raw"
    """
    # Target
    target = add_classification_target(features)
    valid_idx = target.dropna().index.intersection(features.index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]

    # Time-series split
    X_train_raw, X_test_raw, y_train, y_test = time_series_train_test_split(X, y)

    # Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train_raw, X_test_raw)

    # Sklearn classifiers
    print("\n=== Training sklearn classifiers ===")
    trained_clf = train_classifiers(X_train_sc, y_train.values, tune=tune)

    # Ensemble
    print("\n=== Training ensemble ===")
    ensemble = train_ensemble(X_train_sc, y_train.values)

    # MLP
    print("\n=== Training MLP ===")
    mlp, mlp_history = train_mlp(X_train_sc, y_train.values, epochs=mlp_epochs)

    # LSTM
    print("\n=== Training LSTM ===")
    lstm_model, seq_len, lstm_history = train_lstm(X_train_sc, y_train.values, epochs=lstm_epochs)

    return {
        "classifiers": trained_clf,
        "ensemble": ensemble,
        "mlp": mlp,
        "mlp_history": mlp_history,
        "lstm": lstm_model,
        "lstm_history": lstm_history,
        "lstm_seq_len": seq_len,
        "scaler": scaler,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
    }
