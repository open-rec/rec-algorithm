import abc
import math
from collections import defaultdict

from algorithm.recall.recall import EVENT_UNIQUE_COLUMNS, Recall
from algorithm.structure.score_item import ScoreItem


class I2I(Recall):

    def __init__(self, events=None, recall_size=100, cut_size=20):
        super().__init__(events=events, recall_size=recall_size)
        self._cut_size = cut_size

    @abc.abstractmethod
    def dump_i2i(self, cut_size=20):
        pass


class ItemBasedI2I(I2I):
    """
    Item-based collaborative filtering: two items are similar when the same users interacted with
    both. Co-occurrence is damped by the length of the sequence it was observed in, then normalized
    by item popularity so a globally frequent item does not come out similar to everything.
    """

    def __init__(self, events=None, recall_size=100, cut_size=20):
        super().__init__(events=events, recall_size=recall_size, cut_size=cut_size)
        self._similarity = None

    def gen_seq(self):
        # drop_duplicates and sort_values return new frames instead of mutating in place, so their
        # results have to be kept. Order matters here: co-occurrence is read off the behaviour
        # sequence, so leaving the frame unsorted silently changes the outcome.
        events = self._events.drop_duplicates(subset=EVENT_UNIQUE_COLUMNS).sort_values('time')

        user_item_sequence = events.groupby('user_id')[['item_id', 'time']] \
            .apply(lambda x: list(zip(x['item_id'], x['time']))) \
            .reset_index().rename(columns={0: 'item_sequence'})
        return dict(zip(user_item_sequence['user_id'], user_item_sequence['item_sequence']))

    def recall(self, user_triggers=[], item_triggers=[]):
        triggers = item_triggers
        assert triggers

        neighbours = self.dump_i2i(cut_size=self._cut_size)

        # Merge across triggers: an item reachable from several of them keeps its strongest score
        # rather than being emitted once per trigger. Triggers themselves are dropped — the user
        # just acted on them.
        merged = {}
        trigger_set = set(triggers)
        for trigger in triggers:
            for item, score in neighbours.get(trigger, []):
                if item in trigger_set:
                    continue
                if score > merged.get(item, 0):
                    merged[item] = score

        ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[:self._recall_size]
        return [ScoreItem(item=item, score=score) for item, score in ranked]

    def dump_i2i(self, cut_size=20):
        similarity = self._compute_similarity()
        return {
            left_item: sorted(related.items(), key=lambda kv: kv[1], reverse=True)[:cut_size]
            for left_item, related in similarity.items()
        }

    def _compute_similarity(self):
        """
        The expensive half: quadratic in the length of each user's sequence. Cached so that recall()
        and repeated dump_i2i() calls do not pay for it twice.
        """
        if self._similarity is not None:
            return self._similarity

        user_item_time_dict = self.gen_seq()
        i2i_sim = {}
        item_cnt = defaultdict(int)

        for item_time_list in user_item_time_dict.values():
            # a long sequence is weaker evidence that any two of its items belong together
            sequence_weight = 1 / math.log(len(item_time_list) + 1)
            for i, _ in item_time_list:
                item_cnt[i] += 1
                related = i2i_sim.setdefault(i, {})
                for j, _ in item_time_list:
                    if i == j:
                        continue
                    related[j] = related.get(j, 0) + sequence_weight

        for i, related in i2i_sim.items():
            for j in related:
                related[j] /= math.sqrt(item_cnt[i] * item_cnt[j])

        self._similarity = i2i_sim
        return i2i_sim
