"""Testes da Strategy de amostragem negativa."""

import pandas as pd

from recommender.training.sampling import RandomNegativeSampling


def test_negative_sampler_avoids_known_items() -> None:
    positives = pd.DataFrame({"user_id": [0, 0, 1], "item_id": [0, 1, 2]})
    users, items, labels = RandomNegativeSampling(1, seed=42).sample(positives, num_items=5)
    known = {0: {0, 1}, 1: {2}}
    negatives = zip(users[len(positives) :], items[len(positives) :], strict=True)
    assert all(item not in known[user] for user, item in negatives)
    assert labels.sum() == len(positives)
