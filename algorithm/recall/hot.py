import pandas as pd

from algorithm.recall.recall import Recall
from algorithm.structure.score_item import ScoreItem


class Hot(Recall):

    def __init__(self, behaviors=None, hot_size=1000):
        super().__init__(behaviors=behaviors)
        self.hot_size = hot_size

    def recall(self):
        self._behaviors.drop_duplicates((['id', 'user_id', 'item_id', 'time', 'type', 'value']))
        top_i = self._behaviors['item_id'].value_counts().index[:self.hot_size]
        top_v = self._behaviors['item_id'].value_counts().values[:self.hot_size]
        results = []
        for item, value in zip(top_i, top_v):
            results.append(ScoreItem(item=item, score=value * 1.0 / top_v.max()))
        return results


if __name__ == "__main__":
    behaviors = pd.read_csv('../../data/behavior.csv', header=0)
    hot = Hot(behaviors=behaviors, hot_size=100)
    recall_items = hot.recall()
    for item in recall_items:
        print(item)
