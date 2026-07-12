"""Metricas top-K para uma interacao relevante por usuario."""

import numpy as np


def relevance_rank(recommendations: np.ndarray, relevant_item: int) -> int | None:
    """Retorna a posicao base zero do item relevante, quando recomendado."""
    matches = np.flatnonzero(recommendations == relevant_item)
    return int(matches[0]) if len(matches) else None


def user_metrics(recommendations: np.ndarray, relevant_item: int, k: int) -> dict[str, float]:
    """Calcula cinco metricas para um usuario."""
    rank = relevance_rank(recommendations[:k], relevant_item)
    hit = float(rank is not None)
    return {
        "precision_at_k": hit / k,
        "recall_at_k": hit,
        "hit_rate_at_k": hit,
        "ndcg_at_k": 0.0 if rank is None else 1.0 / np.log2(rank + 2),
        "map_at_k": 0.0 if rank is None else 1.0 / (rank + 1),
    }


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Calcula a media de cada metrica entre usuarios."""
    keys = rows[0]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
