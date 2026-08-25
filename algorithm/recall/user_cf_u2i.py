import math
from collections import defaultdict

from algorithm.recall.recall import EVENT_UNIQUE_COLUMNS, Recall
from algorithm.structure.score_item import ScoreItem


class UserBasedCF(Recall):
    """User-based collaborative filtering with inverse item-popularity weighting."""

    def __init__(self, events=None, recall_size=100, neighbour_size=50):
        super().__init__(events=events, recall_size=recall_size)
        self._neighbour_size = neighbour_size
        self._user_items = None
        self._similarity = None

    def _compute_similarity(self):
        if self._similarity is not None:
            return self._similarity

        columns = [column for column in EVENT_UNIQUE_COLUMNS if column in self._events.columns]
        events = self._events.drop_duplicates(subset=columns)
        user_items = events.groupby('user_id')['item_id'].apply(set).to_dict()
        item_users = events.groupby('item_id')['user_id'].apply(set).to_dict()
        similarity = defaultdict(lambda: defaultdict(float))

        for users in item_users.values():
            # A niche item shared by two users is stronger evidence than a globally popular item.
            weight = 1.0 / math.log(len(users) + 1.0)
            for left in users:
                for right in users:
                    if left != right:
                        similarity[left][right] += weight

        for left, related in similarity.items():
            for right in related:
                related[right] /= math.sqrt(len(user_items[left]) * len(user_items[right]))

        self._user_items = user_items
        self._similarity = similarity
        return similarity

    def dump_user_recall(self, neighbour_size=None):
        """Return candidates keyed by user, suitable for the distributed user recall schema."""
        similarity = self._compute_similarity()
        neighbour_size = neighbour_size or self._neighbour_size
        result = {}
        for user, related in similarity.items():
            seen = self._user_items[user]
            candidates = defaultdict(float)
            neighbours = sorted(related.items(), key=lambda row: (-row[1], str(row[0]))) \
                [:neighbour_size]
            for neighbour, score in neighbours:
                for item in self._user_items[neighbour] - seen:
                    candidates[item] += score
            result[user] = sorted(candidates.items(), key=lambda row: (-row[1], str(row[0]))) \
                [:self._recall_size]
        return result

    def recall(self, user_triggers=[], item_triggers=[]):
        assert user_triggers
        tables = self.dump_user_recall()
        merged = defaultdict(float)
        for user in user_triggers:
            for item, score in tables.get(user, []):
                merged[item] += score
        ranked = sorted(merged.items(), key=lambda row: (-row[1], str(row[0]))) \
            [:self._recall_size]
        return [ScoreItem(item=item, score=score) for item, score in ranked]
