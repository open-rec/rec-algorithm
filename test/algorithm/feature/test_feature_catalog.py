import json

import numpy as np
import pandas as pd
import pytest

from algorithm.feature.event_feature import event_feature_columns
from algorithm.feature.feature_catalog import FeatureCatalog, ModelFeatureSet
from algorithm.feature.feature_space import FeatureSpace


def frames():
    users = pd.DataFrame(
        [
            {
                "id": "u1",
                "country": "CN",
                "city": "HZ",
                "gender": 1,
                "age": 30,
                "tags": "sports",
                "event_count": 2,
            }
        ]
    )
    items = pd.DataFrame(
        [
            {
                "id": "i1",
                "category": "sports",
                "scene": "home",
                "weight": 1,
                "event_count": 3,
            }
        ]
    )
    return users, items


def test_catalog_covers_the_realtime_event_feature_contract():
    catalog = FeatureCatalog.load()
    assert {
        catalog.require("user." + name)["column"]
        for name in event_feature_columns("item")
    } == set(event_feature_columns("item"))
    assert {
        catalog.require("item." + name)["column"]
        for name in event_feature_columns("user")
    } == set(event_feature_columns("user"))


@pytest.mark.parametrize(
    "model_type,expected_name",
    [
        ("lr", "ranking-lr-v1"),
        ("fm", "ranking-fm-v1"),
    ],
)
def test_model_feature_sets_fit_and_persist_model_metadata(
    tmp_path, model_type, expected_name
):
    users, items = frames()
    space = FeatureSpace.for_model(model_type).fit(users, items)
    path = tmp_path / (model_type + ".features.json")
    space.save(path)
    payload = json.loads(path.read_text())
    loaded = FeatureSpace.load(path)

    assert payload["feature_set"] == expected_name
    assert payload["model_type"] == model_type
    assert payload["catalog_version"] == 3
    assert len(payload["catalog_sha256"]) == 64
    assert payload["input_dim"] == loaded.dim
    assert loaded.user_columns[0].feature_id == "user.country"
    assert any(
        column.feature_id == "item.event_click_rate"
        for column in loaded.item_columns
    )


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
    monkeypatch.setattr(
        ModelFeatureSet,
        "for_model",
        classmethod(
            lambda cls, model_type: (_ for _ in ()).throw(
                AssertionError("deployment must not consult the feature set")
            )
        ),
    )

    loaded = FeatureSpace.load(path)

    assert loaded.model_type == "fm"
    assert loaded.feature_set == "ranking-fm-v1"


def test_serving_lists_match_training_and_bound_outliers():
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


def test_user_rank_uses_the_user_contract_on_both_sides():
    users, _ = frames()
    space = FeatureSpace.for_model("lr", target_type="user").fit(users, users)
    assert space.target_type == "user"
    assert space.user_width == space.item_width
    assert all(
        column.feature_id.startswith("user.") for column in space.item_columns
    )


@pytest.mark.parametrize("model_type", ["lr", "fm"])
@pytest.mark.parametrize("target_type", ["item", "user"])
def test_selected_features_round_trip(model_type, target_type):
    users, items = frames()
    selection = {
        "user": ["user.age"],
        "candidate": [
            "item.weight" if target_type == "item" else "user.country"
        ],
    }
    candidates = items if target_type == "item" else users
    space = FeatureSpace.for_model(model_type, target_type, selection).fit(
        users, candidates
    )
    restored = FeatureSpace.from_dict(space.to_dict())
    assert restored.selection == selection
    assert restored.dim == 2
    assert np.array_equal(
        space.transform_items(candidates), restored.transform_items(candidates)
    )


def test_unrelated_catalog_change_does_not_invalidate_selected_release(
    monkeypatch,
):
    from copy import deepcopy

    users, items = frames()
    space = FeatureSpace.for_model(
        "lr", selection={"user": ["user.age"], "candidate": ["item.weight"]}
    ).fit(users, items)
    payload = deepcopy(FeatureCatalog.load().payload)
    payload["catalog_version"] += 1
    next(
        feature
        for feature in payload["features"]
        if feature["id"] == "user.city"
    )["definition_version"] += 1
    changed = FeatureCatalog(payload)
    changed.sha256 = "different-global-hash"
    monkeypatch.setattr(
        FeatureCatalog, "load", classmethod(lambda cls: changed)
    )
    assert FeatureSpace.from_dict(space.to_dict()).dim == space.dim
    changed.features["user.age"]["definition_version"] += 1
    with pytest.raises(ValueError, match="incompatible feature"):
        FeatureSpace.from_dict(space.to_dict())


@pytest.mark.parametrize(
    "selection",
    [
        {"user": [], "candidate": ["item.weight"]},
        {"user": ["user.age", "user.age"], "candidate": ["item.weight"]},
        {"user": ["item.weight"], "candidate": ["item.weight"]},
        {"user": ["user.unknown"], "candidate": ["item.weight"]},
        {"user": ["user.age"]},
    ],
)
def test_unsupported_selections_are_rejected(selection):
    with pytest.raises(ValueError):
        FeatureSpace.for_model("lr", selection=selection)


def test_feature_selection_loads_from_spark_zip(tmp_path):
    import subprocess
    import sys
    import zipfile
    from pathlib import Path
    import algorithm

    source = Path(algorithm.__file__).parent
    archive = tmp_path / "algorithm.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for relative in (
            "__init__.py",
            "feature/__init__.py",
            "feature/feature_catalog.py",
        ):
            bundle.write(source / relative, "algorithm/" + relative)
        for path in (source / "feature/definitions").glob("*"):
            if path.is_file():
                bundle.write(
                    path, "algorithm/feature/definitions/" + path.name
                )
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from algorithm.feature.feature_catalog import select_features; "
        "assert select_features('lr', 'item', "
        "{'user':['user.age'], 'candidate':['item.weight']})"
        "['candidate'] == ['item.weight']"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", code, str(archive)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
