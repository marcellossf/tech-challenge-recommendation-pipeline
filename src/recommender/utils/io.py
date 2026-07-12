"""Funcoes pequenas para leitura e escrita de artefatos."""

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Le um arquivo YAML e retorna um dicionario."""
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Escreve JSON identado e cria o diretorio de destino."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    """Le um objeto JSON do disco."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
