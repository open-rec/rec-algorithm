import abc


class Recall(abc.ABC):

    def __init__(self, items=None, users=None, events=None, recall_size=100):
        self._items = items
        self._users = users
        self._events = events
        self._recall_size = recall_size

    @abc.abstractmethod
    def recall(self, user_triggers=[], item_triggers=[]):
        pass
