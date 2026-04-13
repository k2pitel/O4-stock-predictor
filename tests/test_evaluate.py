"""Unit tests for evaluate.py — metric computations."""

import numpy as np
import pytest

from evaluate import classification_metrics, forecast_metrics, rmse, mae


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        m = classification_metrics(y_true, y_pred)
        assert m["Accuracy"] == 1.0
        assert m["F1"] == 1.0

    def test_with_probabilities(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        m = classification_metrics(y_true, y_pred, y_prob)
        assert "ROC-AUC" in m
        assert m["ROC-AUC"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        m = classification_metrics(y_true, y_pred)
        assert m["Accuracy"] == 0.0
        assert m["Recall"] == 0.0


class TestForecastMetrics:
    def test_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        m = forecast_metrics(y, y)
        assert m["RMSE"] == 0.0
        assert m["MAE"] == 0.0

    def test_known_error(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == 1.0
        assert rmse(y_true, y_pred) == 1.0
