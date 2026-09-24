import pandas as pd

from algorithm.feature.context_feature import (candidate_interactions,
    materialize_request_context, materialize_session_features)
from algorithm.feature.feature_space import FeatureSpace


def test_dynamic_roles_are_aligned_and_round_trip():
    users = pd.DataFrame([{"id": "u", "age": 20}])
    items = pd.DataFrame([{"id": "i", "weight": 1}])
    sessions = pd.DataFrame([{"id": "s", "event_count": 2}])
    contexts = materialize_request_context(
        {"device_type": "mobile", "request_time": 0}, ["i"])
    interactions = pd.DataFrame([{"user_item_click_count_log": 1.0}])
    space = FeatureSpace.for_model("lr").fit(
        users, items, sessions, contexts, interactions)

    encoded = space.transform_candidates(
        users, items, sessions, contexts, interactions)
    restored = FeatureSpace.from_dict(space.to_dict())

    assert encoded.shape == (1, space.dim)
    assert restored.selection == space.selection
    assert restored.context_width == space.context_width


def test_session_and_interaction_features_are_point_in_time():
    events = pd.DataFrame([
        {"session_id": "s", "item_id": "i", "type": "click", "value": 2,
         "time": 10},
        {"session_id": "s", "item_id": "i", "type": "click", "value": 3,
         "time": 20},
    ])

    session = materialize_session_features(events, 20, "s")
    interaction = candidate_interactions(events, ["i"], 20).iloc[0]

    assert session["event_count"] == 1
    assert session["event_value_sum"] == 2
    assert interaction["user_item_click_count_log"] > 0
