"""Neural Collaborative Filtering baseado em embeddings e MLP."""

import torch
from torch import nn


class NeuralCollaborativeFiltering(nn.Module):
    """Estima afinidade user-item com embeddings concatenados."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.network = self._build_network(embedding_dim * 2, hidden_dims, dropout)
        self.reset_parameters()

    @staticmethod
    def _build_network(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        for output_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)))
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        return nn.Sequential(*layers)

    def reset_parameters(self) -> None:
        """Inicializa embeddings com distribuicao pequena e reproduzivel."""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Retorna logits para pares de usuarios e itens."""
        user_vector = self.user_embedding(users)
        item_vector = self.item_embedding(items)
        features = torch.cat((user_vector, item_vector), dim=1)
        return self.network(features).squeeze(1)
