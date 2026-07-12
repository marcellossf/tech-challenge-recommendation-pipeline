"""Orquestra baseline NMF e dois experimentos NCF no MLflow."""

import shutil
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader

from recommender.config import get_settings
from recommender.models import ModelFactory
from recommender.training.dataset import InteractionDataset
from recommender.training.early_stopping import EarlyStopping
from recommender.training.sampling import RandomNegativeSampling
from recommender.training.trainer import TrainingResult, fit
from recommender.utils.io import read_json, read_yaml, write_json
from recommender.utils.reproducibility import seed_everything


def build_matrix(frame: pd.DataFrame, num_users: int, num_items: int) -> csr_matrix:
    """Cria matriz binaria usuario-item."""
    values = np.ones(len(frame), dtype=np.float32)
    return csr_matrix((values, (frame.user_id, frame.item_id)), shape=(num_users, num_items))


def train_baseline(train: pd.DataFrame, metadata: dict[str, Any], output: Path) -> None:
    """Treina NMF Scikit-Learn e registra o primeiro experimento."""
    with mlflow.start_run(run_name="baseline-nmf"):
        model = ModelFactory.create("nmf", n_components=32, init="nndsvda", random_state=42)
        matrix = build_matrix(train, metadata["num_users"], metadata["num_items"])
        user_factors = model.fit_transform(matrix)
        joblib.dump((model, user_factors), output)
        mlflow.log_params({"model": "NMF", "components": 32})
        mlflow.log_metric("reconstruction_error", float(model.reconstruction_err_))
        mlflow.log_artifact(str(output), artifact_path="model")


def make_loader(
    frame: pd.DataFrame, num_items: int, negatives: int, seed: int, batch_size: int
) -> DataLoader:
    """Cria DataLoader com positivos e negativos aleatorios."""
    sampler = RandomNegativeSampling(negatives, seed)
    users, items, labels = sampler.sample(frame, num_items)
    dataset = InteractionDataset(users, items, labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def model_config(run: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Combina dimensoes do dataset e hiperparametros da run."""
    return {
        "num_users": metadata["num_users"],
        "num_items": metadata["num_items"],
        "embedding_dim": run["embedding_dim"],
        "hidden_dims": run["hidden_dims"],
        "dropout": run["dropout"],
    }


def _run_neural_experiment(
    run: dict[str, Any], train: pd.DataFrame, validation: pd.DataFrame, metadata: dict[str, Any]
) -> tuple[TrainingResult, str]:
    """Treina uma configuracao neural e registra modelo no Registry."""
    settings = get_settings()
    params = read_yaml("params.yaml")["model"]
    seed_everything(settings.random_seed)
    loaders = neural_loaders(train, validation, metadata["num_items"], params)
    with mlflow.start_run(run_name=run["name"]) as active:
        config = model_config(run, metadata)
        model = ModelFactory.create("neural_cf", **config)
        mlflow.log_params({**run, "model": "NeuralCollaborativeFiltering"})
        stopper = EarlyStopping(params["patience"], params["min_delta"])
        result = fit(model, *loaders, params["learning_rate"], params["epochs"], stopper)
        mlflow.log_metrics(
            {"best_val_loss": result.best_loss, "epochs_trained": result.epochs_trained}
        )
        mlflow.pytorch.log_model(
            result.model, "model", registered_model_name=settings.mlflow_model_name
        )
        return result, active.info.run_id


def neural_loaders(
    train: pd.DataFrame, validation: pd.DataFrame, num_items: int, params: dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    """Monta loaders de treino e validacao com seeds distintas."""
    train_loader = make_loader(
        train, num_items, params["negatives_per_positive"], 42, params["batch_size"]
    )
    val_loader = make_loader(validation, num_items, 1, 43, params["batch_size"])
    return train_loader, val_loader


def save_checkpoint(
    result: TrainingResult, config: dict[str, Any], run_id: str, path: Path
) -> None:
    """Persiste pesos, arquitetura e identificador da run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result.model.state_dict(), "config": config, "run_id": run_id}, path)


def train_candidates(
    runs: list[dict[str, Any]],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    metadata: dict[str, Any],
) -> list[tuple[float, Path, str]]:
    """Treina configuracoes e persiste seus checkpoints."""
    candidates = []
    for run in runs:
        result, run_id = _run_neural_experiment(run, train, validation, metadata)
        checkpoint = get_settings().model_dir / f"{run['name']}.pt"
        save_checkpoint(result, model_config(run, metadata), run_id, checkpoint)
        candidates.append((result.best_loss, checkpoint, run_id))
    return candidates


def select_best(candidates: list[tuple[float, Path, str]]) -> tuple[float, str]:
    """Copia o melhor checkpoint e retorna loss e run ID."""
    best_loss, best_checkpoint, run_id = min(candidates, key=lambda item: item[0])
    shutil.copy2(best_checkpoint, get_settings().model_dir / "ncf_best.pt")
    return best_loss, run_id


def main() -> None:
    """Executa o stage `train` e seleciona o melhor checkpoint."""
    settings = get_settings()
    params = read_yaml("params.yaml")
    metadata = read_json(settings.data_dir / "processed" / "metadata.json")
    train = pd.read_csv(settings.data_dir / "processed" / "train.csv")
    validation = pd.read_csv(settings.data_dir / "processed" / "validation.csv")
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    train_baseline(train, metadata, settings.model_dir / "nmf.joblib")
    candidates = train_candidates(params["experiments"]["neural_runs"], train, validation, metadata)
    best_loss, best_run_id = select_best(candidates)
    write_json(
        {"best_val_loss": best_loss, "run_id": best_run_id},
        settings.report_dir / "training_summary.json",
    )
    print(f"Treino concluido: melhor run={best_run_id}, val_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
