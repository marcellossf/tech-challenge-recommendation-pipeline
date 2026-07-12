"""Factory para construcao desacoplada dos modelos."""

from typing import Any

import torch
from sklearn.decomposition import NMF

from recommender.models.neural_cf import NeuralCollaborativeFiltering


class ModelFactory:
    """Cria implementacoes pelo nome definido na configuracao."""

    @staticmethod
    def create(model_type: str, **config: Any) -> torch.nn.Module | NMF:
        """Instancia um modelo neural ou baseline Scikit-Learn."""
        if model_type == "neural_cf":
            return NeuralCollaborativeFiltering(**config)
        if model_type == "nmf":
            return NMF(**config)
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
