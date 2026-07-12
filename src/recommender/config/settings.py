"""Configuracoes carregadas do ambiente."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centraliza configuracoes externas e seus valores padrao seguros."""

    app_env: str = "development"
    random_seed: int = 42
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "movielens-recommender"
    mlflow_model_name: str = "movielens-ncf"
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Retorna uma instancia cacheada das configuracoes."""
    return Settings()
