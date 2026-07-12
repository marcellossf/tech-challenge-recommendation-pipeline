"""Testes das metricas top-K."""

import numpy as np

from recommender.evaluation.metrics import user_metrics


def test_user_metrics_for_first_position_hit() -> None:
    metrics = user_metrics(np.array([7, 8, 9]), relevant_item=7, k=3)
    assert metrics["precision_at_k"] == 1 / 3
    assert metrics["recall_at_k"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["map_at_k"] == 1.0


def test_user_metrics_for_miss() -> None:
    assert all(value == 0.0 for value in user_metrics(np.array([1, 2]), 9, 2).values())
