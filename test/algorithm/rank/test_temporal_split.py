import pandas as pd
from torch.utils.data import Subset

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.feature_space import FeatureSpace
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.lr import LRRecModel


def test_validation_split_holds_out_newest_events():
    users = pd.DataFrame([
        {"id": "u", "country": "CN", "city": "HZ", "gender": 1,
         "age": 20, "tags": "tech"},
    ])
    items = pd.DataFrame([
        {"id": "i", "category": "tech", "scene": "home", "weight": 1},
    ])
    # Deliberately reverse input order; EventDataSet must establish event-time order itself.
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": event_type, "time": event_time}
        for event_time, event_type in [(40, "click"), (30, "expose"),
                                      (20, "click"), (10, "expose")]
    ])
    empty_history = pd.DataFrame()
    model = LRRecModel(UserFeature(users, empty_history),
                       ItemFeature(items, empty_history), events)

    training, validation = model._split(val_ratio=.5)

    assert isinstance(training, Subset)
    assert isinstance(validation, Subset)
    train_times = model.dataset.events.iloc[list(training.indices)]["time"]
    validation_times = model.dataset.events.iloc[list(validation.indices)]["time"]
    assert list(train_times) == [10, 20]
    assert list(validation_times) == [30, 40]


def test_user_validation_split_preserves_both_labels():
    users = pd.DataFrame([
        {"id": value, "country": "CN", "city": "HZ", "gender": 1,
         "age": 20, "tags": "tech"}
        for value in ("u1", "u2", "u3")
    ])
    events = pd.DataFrame([
        {"user_id": "u1", "item_id": candidate, "type": event_type, "time": event_time}
        for candidate, event_type, event_time in (
            ("u2", "click", 10), ("u3", "click", 20),
            ("u2", "expose", 30), ("u3", "expose", 30))
    ])
    model = LRRecModel(UserFeature(users, pd.DataFrame()),
                       UserFeature(users, pd.DataFrame()), events, target_type="user")

    training, validation = model._split(val_ratio=.5)

    assert set(model.dataset.labels.iloc[list(training.indices)]) == {0, 1}
    assert set(model.dataset.labels.iloc[list(validation.indices)]) == {0, 1}


def test_dataset_encodes_aligned_point_in_time_rows_instead_of_latest_id_map():
    users = pd.DataFrame([{"id": "u", "country": "CN", "city": "HZ", "gender": 1,
                           "age": 20, "tags": "tech"}])
    items = pd.DataFrame([{"id": "i", "category": "tech", "scene": "home", "weight": 1}])
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": "expose", "time": 10},
        {"user_id": "u", "item_id": "i", "type": "click", "time": 20},
    ])
    sample_users = pd.DataFrame([
        {**users.iloc[0].to_dict(), "event_count": 1},
        {**users.iloc[0].to_dict(), "event_count": 2},
    ])
    sample_items = pd.DataFrame([
        {**items.iloc[0].to_dict(), "event_count": 1},
        {**items.iloc[0].to_dict(), "event_count": 2},
    ])

    model = LRRecModel(UserFeature(users), ItemFeature(items), events,
                       sample_users=sample_users, sample_items=sample_items)

    first_user, first_item, _ = model.dataset[0]
    second_user, second_item, _ = model.dataset[1]
    assert not first_user.equal(second_user)
    assert not first_item.equal(second_item)


def test_feature_space_is_fitted_only_on_training_time_slice():
    users = pd.DataFrame([
        {"id": "u1", "country": "CN", "city": "train", "gender": 1, "age": 20},
        {"id": "u2", "country": "CN", "city": "future-only", "gender": 1, "age": 90},
    ])
    items = pd.DataFrame([{"id": "i", "category": "c", "scene": "home", "weight": 1}])
    events = pd.DataFrame([
        {"user_id": "u1", "item_id": "i", "type": "click", "time": n}
        for n in range(1, 9)
    ] + [
        {"user_id": "u2", "item_id": "i", "type": "expose", "time": 9},
        {"user_id": "u2", "item_id": "i", "type": "expose", "time": 10},
    ])

    model = LRRecModel(UserFeature(users), ItemFeature(items), events,
                       feature_space=FeatureSpace.for_model("lr"), validation_ratio=.2)
    city = next(column for column in model.dataset.feature_space.user_columns
                if column.name == "city")
    age = next(column for column in model.dataset.feature_space.user_columns
               if column.name == "age")

    assert city.categories == ["train"]
    assert age.mean == 20
