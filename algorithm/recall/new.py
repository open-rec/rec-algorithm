from algorithm.recall.recall import Recall
from algorithm.structure.score_item import ScoreItem


class New(Recall):

    def __init__(self, items=None, recall_size=1000):
        super().__init__(items=items, recall_size=recall_size)
        self.power = 31

    def recall(self, triggers=[]):
        self._items.drop_duplicates((['id']))
        top_v = self._items.sort_values(by=['pub_time'], ascending=False).values[:self._recall_size]
        id_index = self._items.columns.get_loc('id')
        time_index = self._items.columns.get_loc('pub_time')
        _max_time = self._items['pub_time'].values.max()

        results = []
        for record in top_v:
            results.append(
                ScoreItem(item=record[id_index], score=pow((record[time_index] * 1.0 / _max_time), self.power)))
        return results
