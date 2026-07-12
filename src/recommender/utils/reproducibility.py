"""Controle de aleatoriedade para execucoes reproduziveis."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Fixa seeds das bibliotecas usadas no projeto."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
