"""Baixa e extrai o MovieLens 100K da fonte oficial."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

from recommender.config import get_settings

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_dataset(destination: Path) -> Path:
    """Baixa o arquivo compactado quando ele ainda nao existe."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urlretrieve(DATASET_URL, destination)  # noqa: S310
    return destination


def extract_ratings(archive: Path, output: Path) -> Path:
    """Extrai somente o arquivo de avaliacoes necessario ao pipeline."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        output.write_bytes(bundle.read("ml-100k/u.data"))
    return output


def main() -> None:
    """Materializa o dataset bruto em `data/raw/u.data`."""
    raw_dir = get_settings().data_dir / "raw"
    archive = download_dataset(raw_dir / "ml-100k.zip")
    output = extract_ratings(archive, raw_dir / "u.data")
    print(f"Dataset pronto: {output}")


if __name__ == "__main__":
    main()
