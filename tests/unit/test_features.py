"""Testes de split temporal."""

import pandas as pd

from recommender.features.build import temporal_leave_two_out


def test_temporal_split_reserves_latest_items(interactions: pd.DataFrame) -> None:
    train, validation, test = temporal_leave_two_out(interactions)
    assert len(train) == 9
    assert len(validation) == 3
    assert len(test) == 3
    assert test.groupby("user_id")["timestamp"].first().eq(4).all()
    assert validation.groupby("user_id")["timestamp"].first().eq(3).all()
