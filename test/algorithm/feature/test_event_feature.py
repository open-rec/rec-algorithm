import pandas as pd

from algorithm.feature.event_feature import aggregate_event_features, enrich_entity_features


def _events():
    return pd.DataFrame([
        {"user_id": "u1", "item_id": "i1", "scene": "s1", "type": "expose", "value": 1, "time": 100},
        {"user_id": "u1", "item_id": "i1", "scene": "s1", "type": "click", "value": 2, "time": 200},
        {"user_id": "u1", "item_id": "i2", "scene": "s2", "type": "buy", "value": 3, "time": 300},
        {"user_id": "u2", "item_id": "i1", "scene": "s1", "type": "click", "value": "bad", "time": 400},
        # after the snapshot: must not leak into the materialized features
        {"user_id": "u1", "item_id": "i3", "scene": "s1", "type": "click", "value": 9, "time": 500},
    ])


def test_user_event_aggregation_respects_snapshot_and_counts_types():
    row = aggregate_event_features(_events(), "user", as_of_time=400).set_index("user_id").loc["u1"]
    assert row.event_count == 3
    assert row.event_value_sum == 6
    assert row.event_unique_item_count == 2
    assert row.event_unique_scene_count == 2
    assert row.event_click_count == 1
    assert row.event_expose_count == 1
    assert row.event_click_rate == 0.5
    assert row.event_recency_seconds == 100


def test_item_event_aggregation_and_zero_fill_for_unseen_entity():
    items = pd.DataFrame([{"id": "i1"}, {"id": "never-seen"}])
    enriched = enrich_entity_features(items, _events(), "item", as_of_time=400).set_index("id")
    assert enriched.loc["i1", "event_count"] == 3
    assert enriched.loc["i1", "event_unique_user_count"] == 2
    assert enriched.loc["never-seen", "event_count"] == 0
    assert enriched.loc["never-seen", "event_click_rate"] == 0
