from algorithm.recall.recall import Recall
from algorithm.structure.score_item import ScoreItem


class Hot(Recall):

    def __init__(self, behaviors=None, recall_size=1000):
        super().__init__(behaviors=behaviors, recall_size=recall_size)

    def recall(self, triggers=[]):
        self._behaviors.drop_duplicates((['id', 'user_id', 'item_id', 'time', 'type', 'value']))
        top_i = self._behaviors['item_id'].value_counts().index[:self._recall_size]
        top_v = self._behaviors['item_id'].value_counts().values[:self._recall_size]
        results = []
        for item, value in zip(top_i, top_v):
            results.append(ScoreItem(item=item, score=value * 1.0 / top_v.max()))
        return results
