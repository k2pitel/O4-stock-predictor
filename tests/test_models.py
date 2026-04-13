"""Unit tests for models.py — model builders and helpers."""

import numpy as np
import pytest

from models import (
    get_classifiers,
    get_voting_ensemble,
    build_mlp,
    build_lstm,
    create_sequences,
)


class TestGetClassifiers:
    def test_returns_four_models(self):
        clfs = get_classifiers()
        assert len(clfs) == 4
        expected_names = {"LogisticRegression", "RandomForest", "SVM", "XGBoost"}
        assert set(clfs.keys()) == expected_names

    def test_all_have_fit(self):
        for name, clf in get_classifiers().items():
            assert hasattr(clf, "fit"), f"{name} missing fit()"


class TestVotingEnsemble:
    def test_is_soft_voting(self):
        ens = get_voting_ensemble()
        assert ens.voting == "soft"


class TestCreateSequences:
    def test_shape(self):
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        X_seq, y_seq = create_sequences(X, y, seq_len=10)
        assert X_seq.shape == (41, 10, 5)
        assert y_seq.shape == (41,)

    def test_last_label(self):
        X = np.arange(20).reshape(20, 1).astype(float)
        y = np.arange(20)
        X_seq, y_seq = create_sequences(X, y, seq_len=5)
        assert y_seq[-1] == y[-1]


class TestBuildMLP:
    def test_output_shape(self):
        model = build_mlp(input_dim=10)
        # Single output neuron
        assert model.output_shape == (None, 1)


class TestBuildLSTM:
    def test_output_shape(self):
        model = build_lstm(seq_len=30, n_features=5)
        assert model.output_shape == (None, 1)
