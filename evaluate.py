"""
evaluate.py — Evaluation metrics module.

Classification metrics (accuracy, precision, recall, F1, ROC-AUC) and
forecasting metrics (RMSE, MAE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from math import sqrt

from models import create_sequences


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like  (predicted labels)
    y_prob : array-like | None  (predicted probabilities for class 1)

    Returns
    -------
    dict with metric names as keys.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["ROC-AUC"] = float("nan")
    return metrics


def evaluate_classifier(model, X_test, y_test, name: str = "Model") -> dict:
    """Evaluate a single sklearn-style classifier."""
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    m = classification_metrics(y_test, y_pred, y_prob)
    print(f"  {name:25s}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"AUC={m.get('ROC-AUC', float('nan')):.4f}")
    return m


def evaluate_all_classifiers(trained: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate every classifier and return a summary DataFrame."""
    results = {}
    for name, clf in trained.items():
        results[name] = evaluate_classifier(clf, X_test, y_test, name=name)
    return pd.DataFrame(results).T


def evaluate_keras_model(model, X_test, y_test, name: str = "Keras") -> dict:
    """Evaluate a Keras model that outputs sigmoid probabilities."""
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    m = classification_metrics(y_test, y_pred, y_prob)
    print(f"  {name:25s}  Acc={m['Accuracy']:.4f}  F1={m['F1']:.4f}  "
          f"AUC={m.get('ROC-AUC', float('nan')):.4f}")
    return m


def evaluate_lstm(model: object, X_test_sc: np.ndarray,
                  y_test: pd.Series, seq_len: int) -> dict:
    """Evaluate LSTM by creating sequences from the test set."""
    X_seq, y_seq = create_sequences(X_test_sc, y_test.values, seq_len=seq_len)
    if len(X_seq) == 0:
        print("  LSTM: Not enough test data for sequences.")
        return {}
    return evaluate_keras_model(model, X_seq, y_seq, name="LSTM")


# ---------------------------------------------------------------------------
# Forecasting (regression)
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred) -> float:
    return sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def forecast_metrics(y_true, y_pred) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}


from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler