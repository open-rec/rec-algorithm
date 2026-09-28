import numpy as np
import pandas as pd

from algorithm.feature.content_feature import enrich_item_content_features
from algorithm.feature.feature_space import FeatureSpace


def test_content_features_are_point_in_time_and_alias_compatible():
    rows = pd.DataFrame(
        [{"id": "i", "title": "Cold Start News", "pubTime": "3600"}]
    )
    result = enrich_item_content_features(rows, as_of_time=10800)

    assert result.loc[0, "pub_time"] == "3600"
    assert result.loc[0, "content_age_hours"] == 2.0
    assert result.loc[0, "subcategory"] == ""


def test_hashed_title_is_fixed_width_deterministic_and_persisted():
    users = pd.DataFrame([{"id": "u", "event_count": 1}])
    items = pd.DataFrame(
        [{
            "id": "i",
            "title": "Cold Start News",
            "subcategory": "local",
            "tags": "breaking",
            "content_age_hours": 2,
        }]
    )
    selection = {
        "user": ["user.event_count"],
        "candidate": ["item.title", "item.content_age_hours"],
    }
    space = FeatureSpace.for_model("fm", selection=selection).fit(users, items)
    encoded = space.transform_items(items)
    restored = FeatureSpace.from_dict(space.to_dict())

    assert encoded.shape == (1, 33)
    assert np.count_nonzero(encoded[0, :32]) > 0
    np.testing.assert_array_equal(encoded, restored.transform_items(items))


def test_iso_publish_time_uses_seconds_independent_of_pandas_resolution():
    from algorithm.feature.content_feature import enrich_item_content_features
    now = pd.Timestamp("2026-09-20T00:00:00Z").timestamp()
    values = ["2026-09-19T00:00:00Z", str(int(now - 86400)),
              str(int((now - 86400) * 1000)), None, "bad"]
    result = enrich_item_content_features(pd.DataFrame({"pub_time": values}), now)
    assert result.content_age_hours.tolist() == [24, 24, 24, 0, 0]
