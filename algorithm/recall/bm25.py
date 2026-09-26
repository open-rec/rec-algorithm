"""Dependency-free BM25 recall for local experiments and parity checks."""

import math
import re
from collections import Counter

from algorithm.structure.score_item import ScoreItem


_TOKEN = re.compile(r"[\w]+", re.UNICODE)


def tokenize(value):
    """Normalize text into terms; callers may supply pre-tokenized iterables."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(term).lower() for term in value if str(term).strip()]
    return _TOKEN.findall(str(value).lower())


class BM25Recall(object):
    """Okapi BM25 over generic entity text fields.

    ``documents`` maps entity ids to either strings or field dictionaries.
    Field weights make the contract useful across domains without embedding
    domain-specific names in the algorithm.
    """

    def __init__(self, documents, recall_size=100, field_weights=None, k1=1.2, b=0.75):
        self._recall_size = int(recall_size)
        self._k1 = float(k1)
        self._b = float(b)
        if self._k1 < 0 or not 0 <= self._b <= 1:
            raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
        self._weights = dict(field_weights or {})
        self._ids = []
        self._terms = []
        document_frequency = Counter()
        for entity_id, document in documents.items():
            fields = document if isinstance(document, dict) else {"text": document}
            frequencies = Counter()
            for field, value in fields.items():
                weight = float(self._weights.get(field, 1.0))
                if weight <= 0:
                    continue
                for term, count in Counter(tokenize(value)).items():
                    frequencies[term] += count * weight
            self._ids.append(str(entity_id))
            self._terms.append(frequencies)
            document_frequency.update(frequencies.keys())
        self._lengths = [sum(values.values()) for values in self._terms]
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        count = len(self._ids)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def recall(self, query, exclude=None, recall_size=None):
        terms = tokenize(query)
        if not terms:
            return []
        excluded = {str(value) for value in (exclude or ())}
        scored = []
        for entity_id, frequencies, length in zip(self._ids, self._terms, self._lengths):
            if entity_id in excluded:
                continue
            score = 0.0
            normalization = self._k1 * (
                1.0 - self._b + self._b * length / max(self._average_length, 1e-12))
            for term in terms:
                frequency = frequencies.get(term, 0.0)
                if frequency:
                    score += self._idf.get(term, 0.0) * (
                        frequency * (self._k1 + 1.0) / (frequency + normalization))
            if score > 0:
                scored.append(ScoreItem(item=entity_id, score=score))
        size = self._recall_size if recall_size is None else int(recall_size)
        return sorted(scored, key=lambda value: (-value.score, value.item))[:max(0, size)]
