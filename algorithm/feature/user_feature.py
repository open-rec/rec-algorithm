import abc

import numpy as np

from algorithm.feature.feature import id_feature, num_feature, multi_value_feature, bool_feature


class UserFeature(abc.ABC):

    def __init__(self, users=None, events=None):
        self._users = users
        self._events = events

    @property
    def users(self):
        return self._users

    @property
    def raw_id(self):
        return self._users["id"]

    @property
    def id_features(self):
        return np.hstack([
            self.id,
            self.device_id,
            self.name,
            self.country,
            self.city,
            self.phone
        ])

    @property
    def id(self):
        return id_feature(self._users[["id"]])

    @property
    def device_id(self):
        return id_feature(self._users[["device_id"]])

    @property
    def name(self):
        return id_feature(self._users[["name"]])

    @property
    def gender(self):
        return bool_feature(self._users[["gender"]])

    @property
    def age(self):
        return num_feature(self._users[["age"]])

    @property
    def country(self):
        return id_feature(self._users[["country"]])

    @property
    def city(self):
        return id_feature(self._users[["city"]])

    @property
    def phone(self):
        return id_feature(self._users[["phone"]])

    @property
    def tags(self):
        return multi_value_feature(self._users["tags"])

    @property
    def register_time(self):
        return num_feature(self._users[["register_time"]])

    @property
    def login_time(self):
        return num_feature(self._users[["login_time"]])
