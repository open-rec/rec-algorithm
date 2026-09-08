import math
from collections import defaultdict


class UserBasedU2U:
    """User similarity from shared behaviour, weighted against popular items."""

    def __init__(self, events, neighbour_size=100):
        self.events = events
        self.neighbour_size = neighbour_size

    def dump_u2u(self):
        events = self.events.drop_duplicates(subset=['user_id', 'item_id'])
        user_items = events.groupby('user_id')['item_id'].apply(set).to_dict()
        item_users = events.groupby('item_id')['user_id'].apply(set).to_dict()
        dots = defaultdict(lambda: defaultdict(float))
        for users in item_users.values():
            weight = 1.0 / math.log1p(len(users))
            for left in users:
                for right in users:
                    if left != right:
                        dots[left][right] += weight
        result = {}
        for left, related in dots.items():
            rows = [(right, score / math.sqrt(len(user_items[left]) * len(user_items[right])))
                    for right, score in related.items()]
            result[left] = sorted(rows, key=lambda row: (-row[1], str(row[0]))) \
                [:self.neighbour_size]
        return result

    def rows(self, scene='default'):
        return [{'scene': scene, 'left_user': left, 'right_user': right, 'score': score}
                for left, related in self.dump_u2u().items() for right, score in related]
