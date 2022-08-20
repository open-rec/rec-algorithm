import abc


class Recall(abc.ABC):

    def __init__(self, items=None, users=None, behaviors=None):
        self._items = items
        self._users = users
        self._behaviors = behaviors

    @abc.abstractmethod
    def recall(self):
        pass
