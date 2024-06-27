import abc

from algorithm.feature.feature import id_feature, num_feature, bool_feature, multi_value_feature


class ItemFeature(abc.ABC):

    def __init__(self, items=None, events=None):
        self._items = items
        self._events = events

    @property
    def items(self):
        return self._items

    @property
    def raw_id(self):
        return self._items["id"]

    @property
    def id(self):
        return id_feature(self._items[["id"]])

    @property
    def title(self):
        return id_feature(self._items[["title"]])

    @property
    def category(self):
        return id_feature(self._items[["category"]])

    @property
    def tags(self):
        return multi_value_feature(self._items["tags"])

    @property
    def scene(self):
        return id_feature(self._items[["scene"]])

    @property
    def pub_time(self):
        return num_feature(self._items[["pub_time"]])

    @property
    def modify_time(self):
        return num_feature(self._items[["modify_time"]])

    @property
    def expire_time(self):
        return num_feature(self._items[["expire_time"]])

    @property
    def status(self):
        return bool_feature(self._items[["status"]])

    @property
    def weight(self):
        return num_feature(self._items[["weight"]])
