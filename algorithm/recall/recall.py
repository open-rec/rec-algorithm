import abc


class Recall(abc.ABC):

    def __init__(self, items=None, users=None, behaviors=None, recall_size=100):
        self._items = items
        self._users = users
        self._behaviors = behaviors
        self._recall_size = recall_size

    @abc.abstractmethod
    def recall(self, triggers=[]):
        pass
