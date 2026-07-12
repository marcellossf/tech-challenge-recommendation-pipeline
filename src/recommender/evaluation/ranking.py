"""Gera ranking para os modelos NMF e NCF."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from recommender.evaluation.metrics import aggregate_metrics, user_metrics


def known_items(frame: pd.DataFrame) -> dict[int, set[int]]:
    """Agrupa itens conhecidos por usuario."""
    return frame.groupby("user_id")["item_id"].apply(set).to_dict()


def top_unseen(scores: np.ndarray, seen: set[int], k: int) -> np.ndarray:
    """Seleciona os K maiores scores removendo itens de treino."""
    safe_scores = scores.copy()
    safe_scores[list(seen)] = -np.inf
    candidates = np.argpartition(safe_scores, -k)[-k:]
    return candidates[np.argsort(safe_scores[candidates])[::-1]]


def neural_scorer(model: nn.Module, num_items: int) -> Callable[[int], np.ndarray]:
    """Cria funcao que pontua todo o catalogo para um usuario."""
    items = torch.arange(num_items, dtype=torch.long)

    def score(user: int) -> np.ndarray:
        users = torch.full((num_items,), user, dtype=torch.long)
        with torch.no_grad():
            return model(users, items).numpy()

    return score


def evaluate_ranking(
    scorer: Callable[[int], np.ndarray], train: pd.DataFrame, test: pd.DataFrame, k: int
) -> dict[str, float]:
    """Avalia recomendacoes top-K contra o ultimo item temporal."""
    histories = known_items(train)
    rows = []
    for user, item in test[["user_id", "item_id"]].itertuples(index=False):
        recommendations = top_unseen(scorer(user), histories[user], k)
        rows.append(user_metrics(recommendations, item, k))
    return aggregate_metrics(rows)
