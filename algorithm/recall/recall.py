import abc

# Columns that together identify one event row. Used to drop duplicated rows before counting or
# building sequences, so a row repeated in the source data does not inflate the result.
EVENT_UNIQUE_COLUMNS = ['id', 'user_id', 'item_id', 'time', 'type', 'value']


class Recall(abc.ABC):

    def __init__(self, items=None, users=None, events=None, recall_size=100):
        self._items = items
        self._users = users
        self._events = events
        self._recall_size = recall_size

    @abc.abstractmethod
    def recall(self, user_triggers=[], item_triggers=[]):
        pass
