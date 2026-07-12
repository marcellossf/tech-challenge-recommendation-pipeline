"""Fixtures pequenas e deterministicas."""

import pandas as pd
import pytest


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Cria historico temporal com cinco eventos por usuario."""
    rows = []
    for user in range(3):
        for offset in range(5):
            rows.append({"user_id": user, "item_id": user * 5 + offset, "timestamp": offset})
    return pd.DataFrame(rows)
