from algorithm.recall.recall import EVENT_UNIQUE_COLUMNS, Recall
from algorithm.structure.score_item import ScoreItem


class Hot(Recall):
    """Popularity recall: interaction count per item, scaled so the most popular one scores 1."""

    def __init__(self, events=None, recall_size=1000):
        super().__init__(events=events, recall_size=recall_size)

    def recall(self, user_triggers=[], item_triggers=[]):
        # drop_duplicates returns a new frame rather than mutating in place, so its result has to be
        # kept — otherwise duplicated rows inflate the counts below.
        events = self._events.drop_duplicates(subset=EVENT_UNIQUE_COLUMNS)

        # value_counts sorts descending already, and computing it once is enough
        counts = events['item_id'].value_counts()
        if counts.empty:
            return []

        max_count = counts.iloc[0]
        return [
            ScoreItem(item=item, score=count * 1.0 / max_count)
            for item, count in counts.iloc[:self._recall_size].items()
        ]
