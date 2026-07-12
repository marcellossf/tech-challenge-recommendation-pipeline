"""Dataset PyTorch para exemplos user-item."""

import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionDataset(Dataset[tuple[torch.Tensor, ...]]):
    """Armazena pares user-item e labels binarios."""

    def __init__(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray) -> None:
        self.users = torch.as_tensor(users, dtype=torch.long)
        self.items = torch.as_tensor(items, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        """Retorna o numero de exemplos."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Retorna usuario, item e label de um exemplo."""
        return self.users[index], self.items[index], self.labels[index]
