"""Smoke test de treinamento neural de ponta a ponta."""

import mlflow
import numpy as np
from torch.utils.data import DataLoader

from recommender.models.neural_cf import NeuralCollaborativeFiltering
from recommender.training.dataset import InteractionDataset
from recommender.training.early_stopping import EarlyStopping
from recommender.training.trainer import fit


def test_neural_training_smoke(tmp_path) -> None:
    users = np.array([0, 0, 1, 1])
    items = np.array([0, 2, 1, 3])
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    loader = DataLoader(InteractionDataset(users, items, labels), batch_size=2)
    model = NeuralCollaborativeFiltering(2, 4, 2, [4], 0.0)
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment("smoke-test")
    with mlflow.start_run():
        result = fit(model, loader, loader, 0.01, 2, EarlyStopping(2, 0.0))
    assert result.epochs_trained >= 1
    assert result.best_loss < float("inf")
