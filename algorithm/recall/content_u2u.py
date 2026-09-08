import math
import re
from collections import Counter, defaultdict


class ContentBasedU2U:
    """TF-IDF cosine similarity over categorical user profile attributes."""

    DEFAULT_COLUMNS = ('gender', 'country', 'city', 'tags')

    def __init__(self, users, neighbour_size=100, columns=None):
        self.users = users
        self.neighbour_size = neighbour_size
        self.columns = tuple(columns or self.DEFAULT_COLUMNS)

    def _tokens(self, row):
        tokens = set()
        for column in self.columns:
            if column not in row.index or row[column] is None:
                continue
            value = str(row[column]).strip().lower()
            if not value or value == 'nan':
                continue
            tokens.update('%s:%s' % (column, part) for part in re.split(r'[,|/\s]+', value) if part)
        return tokens

    def dump_u2u(self):
        documents = {row['id']: self._tokens(row)
                     for _, row in self.users.drop_duplicates(subset=['id']).iterrows()}
        documents = {user: tokens for user, tokens in documents.items() if tokens}
        frequencies = Counter(token for tokens in documents.values() for token in tokens)
        count = len(documents)
        vectors = {user: {token: math.log((count + 1.0) / (frequencies[token] + 1.0)) + 1.0
                          for token in tokens} for user, tokens in documents.items()}
        norms = {user: math.sqrt(sum(value * value for value in vector.values()))
                 for user, vector in vectors.items()}
        inverted = defaultdict(list)
        for user, vector in vectors.items():
            for token, value in vector.items():
                inverted[token].append((user, value))
        dots = defaultdict(lambda: defaultdict(float))
        for entries in inverted.values():
            for left, left_value in entries:
                for right, right_value in entries:
                    if left != right:
                        dots[left][right] += left_value * right_value
        return {left: sorted(((right, dot / (norms[left] * norms[right]))
                              for right, dot in related.items()),
                             key=lambda row: (-row[1], str(row[0])))[:self.neighbour_size]
                for left, related in dots.items()}

    def rows(self, scene='default'):
        return [{'scene': scene, 'left_user': left, 'right_user': right, 'score': score}
                for left, related in self.dump_u2u().items() for right, score in related]
