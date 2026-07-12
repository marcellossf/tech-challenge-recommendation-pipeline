"""Estrategias de amostragem para feedback implicito."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SamplingStrategy(ABC):
    """Contrato para estrategias de geracao de exemplos."""

    @abstractmethod
    def sample(self, positives: pd.DataFrame, num_items: int) -> tuple[np.ndarray, ...]:
        """Transforma interacoes positivas em pares rotulados."""


class RandomNegativeSampling(SamplingStrategy):
    """Gera negativos aleatorios que nao aparecem no historico do usuario."""

    def __init__(self, negatives_per_positive: int, seed: int) -> None:
        self.negatives_per_positive = negatives_per_positive
        self.rng = np.random.default_rng(seed)

    def sample(self, positives: pd.DataFrame, num_items: int) -> tuple[np.ndarray, ...]:
        """Retorna vetores de usuario, item e rotulo."""
        users = positives["user_id"].to_numpy(dtype=np.int64)
        items = positives["item_id"].to_numpy(dtype=np.int64)
        known = positives.groupby("user_id")["item_id"].apply(set).to_dict()
        neg_users = np.repeat(users, self.negatives_per_positive)
        neg_items = self._draw_negatives(neg_users, known, num_items)
        labels = np.concatenate((np.ones(len(users)), np.zeros(len(neg_users))))
        return np.concatenate((users, neg_users)), np.concatenate((items, neg_items)), labels

    def _draw_negatives(
        self, users: np.ndarray, known: dict[int, set[int]], num_items: int
    ) -> np.ndarray:
        negatives = np.empty(len(users), dtype=np.int64)
        for index, user in enumerate(users):
            candidate = int(self.rng.integers(num_items))
            while candidate in known[int(user)]:
                candidate = int(self.rng.integers(num_items))
            negatives[index] = candidate
        return negatives
