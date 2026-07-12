"""Cria splits temporais sem vazamento por usuario."""

from pathlib import Path

import pandas as pd

from recommender.config import get_settings
from recommender.utils.io import write_json


def temporal_leave_two_out(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserva as duas interacoes positivas mais recentes por usuario."""
    ordered = interactions.sort_values(["user_id", "timestamp"])
    position = ordered.groupby("user_id").cumcount(ascending=False)
    test = ordered.loc[position == 0].copy()
    validation = ordered.loc[position == 1].copy()
    train = ordered.loc[position >= 2].copy()
    return train, validation, test


def save_split(frame: pd.DataFrame, path: Path) -> None:
    """Salva somente colunas necessarias para modelagem e auditoria."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[["user_id", "item_id", "timestamp"]].to_csv(path, index=False)


def split_summary(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> dict[str, int]:
    """Resume dimensoes e tamanhos dos splits."""
    all_frames = pd.concat([train, validation, test])
    return {
        "num_users": int(all_frames["user_id"].max() + 1),
        "num_items": int(all_frames["item_id"].max() + 1),
        "train_interactions": len(train),
        "validation_interactions": len(validation),
        "test_interactions": len(test),
    }


def main() -> None:
    """Executa o stage `feature_eng` do DVC."""
    settings = get_settings()
    source = settings.data_dir / "interim" / "interactions.csv"
    interactions = pd.read_csv(source)
    train, validation, test = temporal_leave_two_out(interactions)
    processed = settings.data_dir / "processed"
    save_split(train, processed / "train.csv")
    save_split(validation, processed / "validation.csv")
    save_split(test, processed / "test.csv")
    write_json(split_summary(train, validation, test), processed / "metadata.json")
    print(f"Splits prontos: train={len(train)}, validation={len(validation)}, test={len(test)}")


if __name__ == "__main__":
    main()
