"""Loop de treino curto e testavel para o NCF."""

from copy import deepcopy
from dataclasses import dataclass

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from recommender.training.early_stopping import EarlyStopping


@dataclass
class TrainingResult:
    """Resultado minimo necessario para selecao do modelo."""

    model: nn.Module
    best_loss: float
    epochs_trained: int


def batch_loss(
    model: nn.Module, batch: tuple[torch.Tensor, ...], loss_fn: nn.Module
) -> torch.Tensor:
    """Calcula a loss de um lote."""
    users, items, labels = batch
    return loss_fn(model(users, items), labels)


def run_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer
) -> float:
    """Executa uma epoca de otimizacao."""
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = batch_loss(model, batch, loss_fn)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch[0])
    return total / len(loader.dataset)


def validation_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    """Calcula loss media sem atualizar pesos."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            total += batch_loss(model, batch, loss_fn).item() * len(batch[0])
    return total / len(loader.dataset)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    stopper: EarlyStopping,
) -> TrainingResult:
    """Treina com Adam, early stopping e logging por epoca."""
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return _train_until_stopping(
        model, train_loader, validation_loader, loss_fn, optimizer, epochs, stopper
    )


def _train_until_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    stopper: EarlyStopping,
) -> TrainingResult:
    """Executa epocas e restaura os melhores pesos."""
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = validation_loss(model, validation_loader, loss_fn)
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        if val_loss <= stopper.best_loss:
            best_state = deepcopy(model.state_dict())
        if stopper.update(val_loss):
            break
    model.load_state_dict(best_state)
    return TrainingResult(model, stopper.best_loss, epoch)
