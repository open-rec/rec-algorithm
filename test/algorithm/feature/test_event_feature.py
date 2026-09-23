import json
from pathlib import Path

import pandas as pd

from algorithm.feature.event_feature import aggregate_event_features, enrich_entity_features
from algorithm.feature.commerce_feature import aggregate_user_commerce_features
from algorithm.feature.point_in_time import resolve_event_mutations_as_of


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
    assert row.event_ctr == 1.0
    assert row.event_buy_per_click == 1.0
    assert row.event_buy_per_collect == 0.0
    assert row.event_ctr_1d == 1.0
    assert row.event_recency_seconds == 100
    assert row.event_expose_count_5m == 1
    assert row.event_expose_count_1h == 1
    assert row.event_value_sum_5m == 6


def test_item_event_aggregation_and_zero_fill_for_unseen_entity():
    items = pd.DataFrame([{"id": "i1"}, {"id": "never-seen"}])
    enriched = enrich_entity_features(items, _events(), "item", as_of_time=400).set_index("id")
    assert enriched.loc["i1", "event_count"] == 3
    assert enriched.loc["i1", "event_unique_user_count"] == 2
    assert enriched.loc["never-seen", "event_count"] == 0
    assert enriched.loc["never-seen", "event_click_rate"] == 0
    assert enriched.loc["i1", "event_value_sum_5m"] == 3
    assert enriched.loc["never-seen", "event_expose_count_24h"] == 0


def test_short_windows_are_point_in_time_and_filter_exposures():
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": "expose", "value": 2, "time": 100},
        {"user_id": "u", "item_id": "i", "type": "click", "value": 3, "time": 350},
        {"user_id": "u", "item_id": "i", "type": "expose", "value": 5, "time": 401},
    ])
    row = aggregate_event_features(events, "item", as_of_time=400).iloc[0]
    assert row.event_expose_count_5m == 1
    assert row.event_value_sum_5m == 5
    assert row.event_expose_count_1h == 1


def test_event_identity_deduplicates_and_empty_scene_is_missing():
    events = pd.DataFrame([
        {"event_id": "same", "trace_id": "request", "user_id": "u", "item_id": "i", "scene": " ",
         "type": "click", "value": 2, "time": 100},
        {"event_id": "same", "trace_id": "request", "user_id": "u", "item_id": "i", "scene": "ignored",
         "type": "click", "value": 2, "time": 100},
    ])
    row = aggregate_event_features(events, "user", 200).iloc[0]
    assert row.event_count == 1
    assert row.event_unique_scene_count == 0


def test_trace_context_keeps_distinct_expose_and_click_actions():
    events = pd.DataFrame([
        {"trace_id": "request-1", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "expose", "value": 0, "time": 100},
        {"trace_id": "request-1", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "click", "value": 1, "time": 101},
    ])
    row = aggregate_event_features(events, "user", 200).iloc[0]
    assert row.event_count == 2
    assert row.event_expose_count == 1
    assert row.event_click_count == 1
    assert row.event_ctr == 1.0


def test_conversion_rates_use_action_denominators_and_windows():
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "i1", "type": "expose", "time": 1},
        {"user_id": "u", "item_id": "i1", "type": "expose", "time": 2},
        {"user_id": "u", "item_id": "i1", "type": "click", "time": 3},
        {"user_id": "u", "item_id": "i1", "type": "collect", "time": 4},
        {"user_id": "u", "item_id": "i1", "type": "buy", "time": 5},
        {"user_id": "u", "item_id": "i2", "type": "expose", "time": 200000},
    ])
    row = aggregate_event_features(events, "user", 200000).iloc[0]
    assert row.event_ctr == 1 / 3
    assert row.event_collect_per_click == 1.0
    assert row.event_buy_per_click == 1.0
    assert row.event_buy_per_collect == 1.0
    assert row.event_ctr_1d == 0.0
    assert row.event_buy_per_click_1d == 0.0


def test_shared_java_python_golden_fixture():
    path = Path(__file__).parents[3] / "algorithm/feature/definitions/event-feature-parity.json"
    fixture = json.loads(path.read_text())
    events = resolve_event_mutations_as_of(pd.DataFrame(fixture["events"]), 10000)
    row = aggregate_event_features(events, "user",
                                   fixture["as_of_time"]).iloc[0]
    commerce = aggregate_user_commerce_features(
        events, pd.DataFrame(), fixture["as_of_time"]).iloc[0]
    row = pd.concat((row, commerce.drop(labels=["user_id"])))
    for name, expected in fixture["expected_user"].items():
        assert row[name] == expected
    for name, expected in fixture["expected_user_strings"].items():
        assert row[name] == expected
    items = aggregate_event_features(events, "item",
                                     fixture["as_of_time"]).set_index("item_id")
    for item_id, expected_values in fixture["expected_items"].items():
        for name, expected in expected_values.items():
            assert items.loc[item_id, name] == expected
