"""Promove o melhor modelo registrado para Staging e Production."""

from mlflow import MlflowClient

from recommender.config import get_settings
from recommender.utils.io import read_json


def version_for_run(client: MlflowClient, model_name: str, run_id: str) -> str:
    """Localiza a versao do Registry criada por uma run."""
    versions = client.search_model_versions(f"name='{model_name}'")
    matches = [version for version in versions if version.run_id == run_id]
    if not matches:
        raise RuntimeError(f"Nenhuma versao encontrada para run {run_id}")
    return max(matches, key=lambda version: int(version.version)).version


def main() -> None:
    """Promove a versao selecionada e adiciona o alias atual."""
    settings = get_settings()
    summary = read_json(settings.report_dir / "training_summary.json")
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = version_for_run(client, settings.mlflow_model_name, summary["run_id"])
    client.transition_model_version_stage(settings.mlflow_model_name, version, "Staging")
    client.transition_model_version_stage(
        settings.mlflow_model_name, version, "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(settings.mlflow_model_name, "champion", version)
    print(f"Modelo {settings.mlflow_model_name} v{version} promovido para Production")


if __name__ == "__main__":
    main()
