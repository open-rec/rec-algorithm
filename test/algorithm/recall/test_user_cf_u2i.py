import math

import pandas as pd

from algorithm.recall.user_cf_u2i import UserBasedCF


def events():
    return pd.DataFrame([
        ("e1", "u1", "a", 1, "click", "1"),
        ("e2", "u1", "b", 2, "click", "1"),
        ("e3", "u2", "a", 3, "click", "1"),
        ("e4", "u2", "c", 4, "click", "1"),
        ("e5", "u3", "b", 5, "click", "1"),
        ("e6", "u3", "d", 6, "click", "1"),
    ], columns=["id", "user_id", "item_id", "time", "type", "value"])


def test_user_cf_recalls_unseen_items_from_similar_users():
    model = UserBasedCF(events=events(), recall_size=10)
    results = {result.item: result.score for result in model.recall(user_triggers=["u1"])}

    assert set(results) == {"c", "d"}
    assert math.isclose(results["c"], results["d"])


def test_user_cf_deduplicates_events_and_never_returns_seen_items():
    source = pd.concat([events(), events().iloc[[0]]], ignore_index=True)
    results = UserBasedCF(events=source, recall_size=10).dump_user_recall()["u1"]

    assert {item for item, _ in results}.isdisjoint({"a", "b"})
