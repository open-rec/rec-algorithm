import abc
import math
from collections import defaultdict

from algorithm.recall.recall import Recall
from algorithm.structure.score_item import ScoreItem


class I2I(Recall):

    def __init__(self, events=None, recall_size=100, cut_size=20):
        super().__init__(events=events, recall_size=recall_size)
        self._cut_size = cut_size

    @abc.abstractmethod
    def dump_i2i(self, i2i_size=20):
        pass


class ItemBasedI2I(I2I):

    def __init__(self, events=None, recall_size=100, cut_size=20):
        super().__init__(events=events, recall_size=recall_size, cut_size=cut_size)

    def gen_seq(self):
        def make_item_time_pair(df):
            return list(zip(df['item_id'], df['time']))

        self._events.drop_duplicates((['id', 'user_id', 'item_id', 'time', 'type', 'value']))
        self._events.sort_values('time')
        user_item_sequence = self._events.groupby('user_id')[['item_id', 'time']] \
            .apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_sequence'})
        return dict(zip(user_item_sequence['user_id'], user_item_sequence['item_sequence']))

    def recall(self, user_triggers=[], item_triggers=[]):
        triggers = item_triggers
        assert triggers

        cut_full_i2i_items = self.dump_i2i(cut_size=self._cut_size)
        recall_items = []
        inner_size = math.ceil(self._recall_size / len(triggers))
        remain = self._recall_size
        for trigger in triggers:
            if trigger in cut_full_i2i_items and cut_full_i2i_items[trigger]:
                sort_items = cut_full_i2i_items[trigger]
                for item, score in sort_items:
                    recall_items.append(ScoreItem(item=item, score=score))
                remain -= inner_size
        return recall_items

    def dump_i2i(self, cut_size=20):
        user_item_time_dict = self.gen_seq()
        i2i_sim = {}
        item_cnt = defaultdict(int)
        for user, item_time_list in user_item_time_dict.items():
            for i, i_click_time in item_time_list:
                item_cnt[i] += 1
                i2i_sim.setdefault(i, {})
                for j, j_click_time in item_time_list:
                    if i == j:
                        continue
                    i2i_sim[i].setdefault(j, 0)
                    i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

        full_i2i_items = i2i_sim.copy()
        for i, related_items in i2i_sim.items():
            for j, wij in related_items.items():
                full_i2i_items[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])
        full_i2i_items
        cut_full_i2i_items = {
            left_item: sorted(full_i2i_items[left_item].items(), key=lambda item: item[1], reverse=True)[
                       :cut_size]
            for left_item in full_i2i_items
        }
        return cut_full_i2i_items
