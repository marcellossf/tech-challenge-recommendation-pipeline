"""Testes de early stopping."""

from recommender.training.early_stopping import EarlyStopping


def test_early_stopping_counts_only_stale_epochs() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.update(1.0) is False
    assert stopper.update(0.8) is False
    assert stopper.update(0.805) is False
    assert stopper.update(0.81) is True
