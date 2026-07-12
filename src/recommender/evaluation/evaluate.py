"""Compara NMF e NCF nas mesmas metricas de ranking."""

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch

from recommender.config import get_settings
from recommender.evaluation.ranking import evaluate_ranking, neural_scorer
from recommender.models import ModelFactory
from recommender.utils.io import read_json, read_yaml, write_json


def load_neural(path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Reconstrui modelo neural a partir do checkpoint selecionado."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ModelFactory.create("neural_cf", **checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def nmf_scorer(model: Any, user_factors: np.ndarray) -> Any:
    """Cria pontuador baseado na reconstrucao NMF."""
    return lambda user: user_factors[user] @ model.components_


def comparison_markdown(results: dict[str, dict[str, float]], k: int) -> str:
    """Gera uma tabela Markdown para comparar os modelos."""
    headers = list(next(iter(results.values())))
    lines = [f"# Comparacao de modelos (K={k})", "", "| Modelo | " + " | ".join(headers) + " |"]
    lines.append("|---|" + "---:|" * len(headers))
    for name, metrics in results.items():
        values = " | ".join(f"{metrics[key]:.4f}" for key in headers)
        lines.append(f"| {name} | {values} |")
    return "\n".join(lines) + "\n"


def evaluate_models(
    train: pd.DataFrame, test: pd.DataFrame, num_items: int, k: int
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Carrega os modelos e calcula metricas comparaveis."""
    settings = get_settings()
    nmf, factors = joblib.load(settings.model_dir / "nmf.joblib")
    neural, checkpoint = load_neural(settings.model_dir / "ncf_best.pt")
    results = {
        "NMF (Scikit-Learn)": evaluate_ranking(nmf_scorer(nmf, factors), train, test, k),
        "NCF (PyTorch)": evaluate_ranking(neural_scorer(neural, num_items), train, test, k),
    }
    return results, checkpoint


def log_evaluation(results: dict[str, dict[str, float]], checkpoint: dict[str, Any]) -> None:
    """Anexa metricas e tabela a run neural vencedora."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    metrics = {f"test_{key}": value for key, value in results["NCF (PyTorch)"].items()}
    with mlflow.start_run(run_id=checkpoint["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(
            str(settings.report_dir / "model_comparison.md"), artifact_path="evaluation"
        )


def main() -> None:
    """Executa o stage `evaluate` e registra metricas finais."""
    settings = get_settings()
    metadata = read_json(settings.data_dir / "processed" / "metadata.json")
    k = read_yaml("params.yaml")["evaluation"]["top_k"]
    train = pd.read_csv(settings.data_dir / "processed" / "train.csv")
    test = pd.read_csv(settings.data_dir / "processed" / "test.csv")
    results, checkpoint = evaluate_models(train, test, metadata["num_items"], k)
    write_json(results, settings.report_dir / "metrics.json")
    report = comparison_markdown(results, k)
    (settings.report_dir / "model_comparison.md").write_text(report, encoding="utf-8")
    log_evaluation(results, checkpoint)
    print(report)


if __name__ == "__main__":
    main()
