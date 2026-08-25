import math
import re
from collections import Counter, defaultdict

from algorithm.recall.item_cf_i2i import I2I
from algorithm.structure.score_item import ScoreItem


DEFAULT_CONTENT_COLUMNS = ('category', 'tags', 'title')
TOKEN_SEPARATOR = r'[,/|\s]+'


class ContentBasedI2I(I2I):
    """TF-IDF cosine item similarity over the existing item content columns."""

    def __init__(self, items=None, recall_size=100, cut_size=20,
                 content_columns=DEFAULT_CONTENT_COLUMNS):
        super().__init__(events=None, recall_size=recall_size, cut_size=cut_size)
        self._items = items
        self._content_columns = tuple(content_columns)
        self._similarity = None

    def _tokens(self, row):
        tokens = set()
        for column in self._content_columns:
            if column not in row.index or row[column] is None:
                continue
            value = str(row[column]).strip().lower()
            if not value or value == 'nan':
                continue
            tokens.update('%s:%s' % (column, token) for token in re.split(TOKEN_SEPARATOR, value)
                          if token)
        return tokens

    def _compute_similarity(self):
        if self._similarity is not None:
            return self._similarity
        items = self._items.drop_duplicates(subset=['id'])
        documents = {row['id']: self._tokens(row) for _, row in items.iterrows()}
        documents = {item: tokens for item, tokens in documents.items() if tokens}
        document_count = len(documents)
        frequencies = Counter(token for tokens in documents.values() for token in tokens)
        vectors = {item: {token: math.log((document_count + 1.0) /
                                          (frequencies[token] + 1.0)) + 1.0
                          for token in tokens} for item, tokens in documents.items()}
        norms = {item: math.sqrt(sum(value * value for value in vector.values()))
                 for item, vector in vectors.items()}
        inverted = defaultdict(list)
        for item, vector in vectors.items():
            for token, value in vector.items():
                inverted[token].append((item, value))
        dots = defaultdict(lambda: defaultdict(float))
        for entries in inverted.values():
            for left, left_value in entries:
                for right, right_value in entries:
                    if left != right:
                        dots[left][right] += left_value * right_value
        self._similarity = {
            left: {right: dot / (norms[left] * norms[right])
                   for right, dot in related.items()}
            for left, related in dots.items()
        }
        return self._similarity

    def dump_i2i(self, cut_size=20):
        return {left: sorted(related.items(), key=lambda row: (-row[1], str(row[0])))[:cut_size]
                for left, related in self._compute_similarity().items()}

    def recall(self, user_triggers=[], item_triggers=[]):
        assert item_triggers
        neighbours = self.dump_i2i(self._cut_size)
        triggers = set(item_triggers)
        merged = defaultdict(float)
        for trigger in item_triggers:
            for item, score in neighbours.get(trigger, []):
                if item not in triggers:
                    merged[item] = max(merged[item], score)
        ranked = sorted(merged.items(), key=lambda row: (-row[1], str(row[0]))) \
            [:self._recall_size]
        return [ScoreItem(item=item, score=score) for item, score in ranked]
