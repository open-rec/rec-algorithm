import re
from functools import cached_property

import numpy as np

from algorithm.feature.feature import id_feature, num_feature, multi_value_feature, bool_feature
from algorithm.feature.feature_space import DEFAULT_MULTI_SEP


class UserFeature(object):
    """
    Ad-hoc encoding of a user frame, one property per column.

    Each property fits its own encoder on `users`, so the layout it produces describes *this frame
    only* — fine for exploration, not safe to persist. Anything that has to agree with a trained
    model (training, serving) should go through `FeatureSpace` instead, which pins the vocabulary
    down and can be saved next to the checkpoint.

    Results are cached per instance: `id_features` alone used to trigger six independent
    `fit_transform` passes over the frame. Build a new instance if `users` changes.
    """

    def __init__(self, users=None, events=None):
        self._users = users
        # accepted for symmetry with ItemFeature and because callers already pass it; behavioural
        # features (CTR, recency, event counts) would be derived from it, none are implemented yet
        self._events = events

    @property
    def users(self):
        return self._users

    @property
    def events(self):
        return self._events

    @property
    def raw_id(self):
        return self._users["id"]

    @cached_property
    def id_features(self):
        return np.hstack([
            self.id,
            self.device_id,
            self.name,
            self.country,
            self.city,
            self.phone
        ])

    @cached_property
    def id(self):
        return id_feature(self._users[["id"]])

    @cached_property
    def device_id(self):
        return id_feature(self._users[["device_id"]])

    @cached_property
    def name(self):
        return id_feature(self._users[["name"]])

    @cached_property
    def gender(self):
        return bool_feature(self._users[["gender"]])

    @cached_property
    def age(self):
        return num_feature(self._users[["age"]])

    @cached_property
    def country(self):
        return id_feature(self._users[["country"]])

    @cached_property
    def city(self):
        return id_feature(self._users[["city"]])

    @cached_property
    def phone(self):
        return id_feature(self._users[["phone"]])

    @cached_property
    def tags(self):
        # split on "," or "/" — item tags used the latter, so both encode the same way now
        return multi_value_feature(self._users["tags"].fillna(""),
                                   tokenizer=lambda x: re.split(DEFAULT_MULTI_SEP, x))

    @cached_property
    def register_time(self):
        return num_feature(self._users[["register_time"]])

    @cached_property
    def login_time(self):
        return num_feature(self._users[["login_time"]])
