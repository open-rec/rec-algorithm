import pandas as pd

from algorithm.feature.point_in_time import materialize_point_in_time_samples


def test_materializes_each_label_without_future_profile_or_behavior_leakage():
    labels = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": "expose", "time": 100},
        {"user_id": "u", "item_id": "i", "type": "click", "time": 200},
    ])
    behavior = pd.DataFrame([
        {"event_id": "past", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "click", "value": "1", "time": 50},
        {"event_id": "between", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "click", "value": "1", "time": 150},
        {"event_id": "same-time", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "click", "value": "1", "time": 200},
    ])
    users = pd.DataFrame([
        {"id": "u", "city": "old", "_effective_time": 10, "_operation": "INSERT"},
        {"id": "u", "city": "future-at-first-label", "_effective_time": 150,
         "_operation": "INSERT"},
    ])
    items = pd.DataFrame([
        {"id": "i", "scene": "s", "category": "c", "_effective_time": 10,
         "_operation": "INSERT"},
    ])

    events, sample_users, sample_items = materialize_point_in_time_samples(
        labels, behavior, users, items)

    assert events["time"].tolist() == [100, 200]
    assert sample_users["city"].tolist() == ["old", "future-at-first-label"]
    assert sample_users["event_count"].tolist() == [1, 2]
    assert sample_items["event_count"].tolist() == [1, 2]


def test_drops_label_when_latest_as_of_entity_version_is_delete():
    labels = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": "expose", "time": 100},
        {"user_id": "u", "item_id": "i", "type": "click", "time": 200},
    ])
    users = pd.DataFrame([{"id": "u", "_effective_time": 0, "_operation": "INSERT"}])
    items = pd.DataFrame([
        {"id": "i", "_effective_time": 0, "_operation": "INSERT"},
        {"id": "i", "_effective_time": 150, "_operation": "DELETE"},
    ])

    events, sample_users, sample_items = materialize_point_in_time_samples(
        labels, pd.DataFrame(), users, items)

    assert events["time"].tolist() == [100]
    assert len(sample_users) == len(sample_items) == 1


def test_event_delete_only_affects_labels_after_its_mutation_time():
    labels = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": "expose", "time": 100},
        {"user_id": "u", "item_id": "i", "type": "click", "time": 200},
    ])
    behavior = pd.DataFrame([
        {"event_id": "e", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "click", "time": 50, "_effective_time": 60, "_operation": "INSERT"},
        {"event_id": "e", "user_id": "u", "item_id": "i", "scene": "s",
         "type": "click", "time": 50, "_effective_time": 150, "_operation": "DELETE"},
    ])
    users = pd.DataFrame([{"id": "u", "_effective_time": 0}])
    items = pd.DataFrame([{"id": "i", "_effective_time": 0}])

    _, sample_users, _ = materialize_point_in_time_samples(labels, behavior, users, items)

    assert sample_users["event_count"].tolist() == [1, 0]
