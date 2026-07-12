"""Converte avaliacoes explicitas em interacoes positivas implicitas."""

from pathlib import Path

import pandas as pd

from recommender.config import get_settings
from recommender.utils.io import read_yaml, write_json

COLUMNS = ["user_raw", "item_raw", "rating", "timestamp"]


def load_ratings(path: Path) -> pd.DataFrame:
    """Le o formato tabular original do MovieLens 100K."""
    return pd.read_csv(path, sep="\t", names=COLUMNS, dtype="int64")


def keep_active_positive_users(
    ratings: pd.DataFrame, minimum_rating: int, minimum_interactions: int
) -> pd.DataFrame:
    """Mantem feedback positivo de usuarios com historico suficiente."""
    positive = ratings.loc[ratings["rating"] >= minimum_rating].copy()
    counts = positive.groupby("user_raw")["item_raw"].transform("size")
    return positive.loc[counts >= minimum_interactions].copy()


def encode_ids(interactions: pd.DataFrame) -> pd.DataFrame:
    """Mapeia IDs esparsos para indices contiguos iniciados em zero."""
    result = interactions.sort_values(["user_raw", "timestamp"]).copy()
    result["user_id"] = pd.factorize(result["user_raw"], sort=True)[0]
    result["item_id"] = pd.factorize(result["item_raw"], sort=True)[0]
    return result[["user_id", "item_id", "timestamp", "user_raw", "item_raw"]]


def dataset_summary(interactions: pd.DataFrame) -> dict[str, int]:
    """Calcula metadados auditaveis do conjunto preprocessado."""
    return {
        "interactions": len(interactions),
        "users": interactions["user_id"].nunique(),
        "items": interactions["item_id"].nunique(),
    }


def main() -> None:
    """Executa o stage `preprocess` do DVC."""
    settings = get_settings()
    params = read_yaml("params.yaml")["data"]
    ratings = load_ratings(settings.data_dir / "raw" / "u.data")
    filtered = keep_active_positive_users(
        ratings, params["positive_rating"], params["min_user_interactions"]
    )
    interactions = encode_ids(filtered)
    destination = settings.data_dir / "interim" / "interactions.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(destination, index=False)
    write_json(dataset_summary(interactions), settings.report_dir / "dataset_summary.json")
    print(f"Preprocessamento concluido: {len(interactions)} interacoes")


if __name__ == "__main__":
    main()
