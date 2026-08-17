import re
from functools import cached_property

from algorithm.feature.feature import (
    id_feature,
    num_feature,
    bool_feature,
    multi_value_feature,
    text_feature,
)
from algorithm.feature.feature_space import DEFAULT_MULTI_SEP


class ItemFeature(object):
    """
    Ad-hoc encoding of an item frame, one property per column. See `UserFeature` — same caveat, the
    layout is a property of the frame it was fitted on, so use `FeatureSpace` for anything a model
    has to agree with. Cached per instance; build a new one if `items` changes.
    """

    def __init__(self, items=None, events=None):
        self._items = items
        # see UserFeature: kept for symmetry, behavioural features are not implemented yet
        self._events = events

    @property
    def items(self):
        return self._items

    @property
    def events(self):
        return self._events

    @property
    def raw_id(self):
        return self._items["id"]

    @cached_property
    def id(self):
        return id_feature(self._items[["id"]])

    @cached_property
    def title(self):
        # single brackets: text_feature needs a Series of strings, not a one-column DataFrame
        return text_feature(self._items["title"].fillna(""))

    @cached_property
    def category(self):
        return multi_value_feature(self._items["category"].fillna(""),
                                   tokenizer=lambda x: re.split(DEFAULT_MULTI_SEP, x))

    @cached_property
    def tags(self):
        return multi_value_feature(self._items["tags"].fillna(""),
                                   tokenizer=lambda x: re.split(DEFAULT_MULTI_SEP, x))

    @cached_property
    def scene(self):
        return id_feature(self._items[["scene"]])

    @cached_property
    def pub_time(self):
        return num_feature(self._items[["pub_time"]])

    @cached_property
    def modify_time(self):
        return num_feature(self._items[["modify_time"]])

    @cached_property
    def expire_time(self):
        return num_feature(self._items[["expire_time"]])

    @cached_property
    def status(self):
        return bool_feature(self._items[["status"]])

    @cached_property
    def weight(self):
        return num_feature(self._items[["weight"]])
