"""Valida dependencias e diretorios necessarios."""

import importlib.util
import sys
from pathlib import Path

from recommender.config import get_settings

REQUIRED_MODULES = ("torch", "sklearn", "mlflow", "dvc", "pandas")


def missing_modules() -> list[str]:
    """Lista bibliotecas obrigatorias que nao podem ser importadas."""
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def ensure_directories(paths: tuple[Path, ...]) -> None:
    """Cria os diretorios de trabalho quando necessario."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Executa validacao reproduzivel do ambiente."""
    settings = get_settings()
    ensure_directories((settings.data_dir, settings.model_dir, settings.report_dir))
    missing = missing_modules()
    if missing:
        raise RuntimeError(f"Dependencias ausentes: {', '.join(missing)}")
    print(f"Ambiente valido | Python {sys.version.split()[0]} | seed={settings.random_seed}")


if __name__ == "__main__":
    main()
