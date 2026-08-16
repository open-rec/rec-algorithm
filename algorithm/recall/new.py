from algorithm.recall.recall import Recall
from algorithm.structure.score_item import ScoreItem


class New(Recall):
    """
    Freshness recall: the most recently published items, scored by how new they are relative to the
    range actually present in the data.
    """

    def __init__(self, items=None, recall_size=1000):
        super().__init__(items=items, recall_size=recall_size)
        # shapes the score curve only; the ordering is the same for any positive power
        self.power = 31

    def recall(self, user_triggers=[], item_triggers=[]):
        # drop_duplicates returns a new frame rather than mutating in place, so keep its result
        items = self._items.drop_duplicates(subset=['id'])
        if items.empty:
            return []

        pub_time = items['pub_time'].astype(float)
        oldest, newest = pub_time.min(), pub_time.max()
        span = newest - oldest

        top = items.assign(_pub_time=pub_time).sort_values('_pub_time', ascending=False) \
            .head(self._recall_size)

        results = []
        for item_id, timestamp in zip(top['id'], top['_pub_time']):
            # Normalize over the observed range. Dividing by the maximum alone leaves every ratio
            # within a hair of 1, because these are absolute epoch seconds — raised to `power` the
            # sample data scored 1.0, 0.9999958, 0.9999933, i.e. no usable spread. When every item
            # shares a timestamp there is no range to speak of, so they are all equally new.
            freshness = 1.0 if span <= 0 else (timestamp - oldest) / span
            results.append(ScoreItem(item=item_id, score=pow(freshness, self.power)))
        return results
