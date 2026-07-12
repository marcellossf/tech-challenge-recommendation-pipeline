"""Testes do design pattern Factory."""

import pytest
from sklearn.decomposition import NMF

from recommender.models import ModelFactory
from recommender.models.neural_cf import NeuralCollaborativeFiltering


def test_factory_creates_both_model_families() -> None:
    baseline = ModelFactory.create("nmf", n_components=2)
    neural = ModelFactory.create(
        "neural_cf", num_users=3, num_items=5, embedding_dim=2, hidden_dims=[4], dropout=0.0
    )
    assert isinstance(baseline, NMF)
    assert isinstance(neural, NeuralCollaborativeFiltering)


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        ModelFactory.create("inexistente")
