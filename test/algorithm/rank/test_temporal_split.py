import pandas as pd
from torch.utils.data import Subset

from algorithm.feature.item_feature import ItemFeature
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
