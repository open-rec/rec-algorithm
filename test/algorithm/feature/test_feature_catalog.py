import json

import numpy as np
import pandas as pd
import pytest

from algorithm.feature.event_feature import event_feature_columns
from algorithm.feature.feature_catalog import FeatureCatalog, ModelFeatureSet
from algorithm.feature.feature_space import FeatureSpace


def frames():
    users = pd.DataFrame([{"id": "u1", "country": "CN", "city": "HZ", "gender": 1,
                           "age": 30, "tags": "sports", "event_count": 2}])
    items = pd.DataFrame([{"id": "i1", "category": "sports", "scene": "home",
                           "weight": 1, "event_count": 3}])
    return users, items


def test_catalog_covers_the_realtime_event_feature_contract():
    catalog = FeatureCatalog.load()
    assert {catalog.require("user." + name)["column"]
            for name in event_feature_columns("item")} == set(event_feature_columns("item"))
    assert {catalog.require("item." + name)["column"]
            for name in event_feature_columns("user")} == set(event_feature_columns("user"))


@pytest.mark.parametrize("model_type,expected_name", [
    ("lr", "ranking-lr-v1"),
    ("fm", "ranking-fm-v1"),
])
def test_model_feature_sets_fit_and_persist_model_metadata(tmp_path, model_type, expected_name):
    users, items = frames()
    space = FeatureSpace.for_model(model_type).fit(users, items)
    path = tmp_path / (model_type + ".features.json")
    space.save(path)
    payload = json.loads(path.read_text())
    loaded = FeatureSpace.load(path)

    assert payload["feature_set"] == expected_name
    assert payload["model_type"] == model_type
    assert payload["catalog_version"] == 1
    assert payload["input_dim"] == loaded.dim
    assert loaded.user_columns[0].feature_id == "user.country"
    assert any(column.feature_id == "item.event_click_rate"
               for column in loaded.item_columns)


def test_lr_and_fm_select_complete_but_independent_feature_sets():
    lr = ModelFeatureSet.for_model("lr")
    fm = ModelFeatureSet.for_model("fm")
    assert lr.name != fm.name
    assert lr.model_type == "lr" and fm.model_type == "fm"
    assert [name for name, _ in lr.user] == [name for name, _ in fm.user]
    assert [name for name, _ in lr.item] == [name for name, _ in fm.item]


def test_fitted_width_metadata_is_validated(tmp_path):
    users, items = frames()
    path = tmp_path / "lr.features.json"
    FeatureSpace.for_model("lr").fit(users, items).save(path)
    payload = json.loads(path.read_text())
    payload["input_dim"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="input_dim"):
        FeatureSpace.load(path)


def test_deployment_loads_only_the_fitted_sidecar(tmp_path, monkeypatch):
    users, items = frames()
    path = tmp_path / "fm.features.json"
    FeatureSpace.for_model("fm").fit(users, items).save(path)
    monkeypatch.setattr(ModelFeatureSet, "for_model",
                        classmethod(lambda cls, model_type: (_ for _ in ()).throw(
                            AssertionError("deployment must not consult the feature set"))))

    loaded = FeatureSpace.load(path)

    assert loaded.model_type == "fm"
    assert loaded.feature_set == "ranking-fm-v1"


def test_serving_lists_match_training_multi_values_and_numeric_outliers_are_bounded():
    users, items = frames()
    users.loc[0, "tags"] = "sports,local"
    users.loc[0, "event_count"] = 2
    space = FeatureSpace.for_model("fm").fit(users, items)
    training = space.transform_users(users)
    serving = users.copy()
    serving["tags"] = pd.Series([["sports", "local"]], dtype=object)
    assert np.array_equal(training, space.transform_users(serving))

    outlier = users.copy()
    outlier.loc[0, "event_count"] = 1000000
    encoded = space.transform_users(outlier)
    assert np.max(np.abs(encoded)) <= 3.0
