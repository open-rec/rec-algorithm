"""Dependency-free BM25 recall for local experiments and parity checks."""

import math
import re
from collections import Counter

import numpy as np

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
        self._postings = {}
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
            position = len(self._ids) - 1
            for term, frequency in frequencies.items():
                self._postings.setdefault(term, []).append((position, frequency))
        self._lengths = [sum(values.values()) for values in self._terms]
        self._index = {entity_id: position for position, entity_id in enumerate(self._ids)}
        self._average_length = sum(self._lengths) / max(len(self._lengths), 1)
        count = len(self._ids)
        self._idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self._postings = {
            term: (np.asarray([position for position, _ in values], dtype=np.int64),
                   np.asarray([frequency for _, frequency in values], dtype=np.float64))
            for term, values in self._postings.items()
        }
        lengths = np.asarray(self._lengths, dtype=np.float64)
        self._normalizations = self._k1 * (
            1.0 - self._b + self._b * lengths / max(self._average_length, 1e-12))

    def recall(self, query, exclude=None, recall_size=None):
        terms = tokenize(query)
        if not terms:
            return []
        excluded = {str(value) for value in (exclude or ())}
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for term, query_frequency in Counter(terms).items():
            idf = self._idf.get(term, 0.0)
            postings = self._postings.get(term)
            if postings is None:
                continue
            positions, frequencies = postings
            scores[positions] += query_frequency * idf * (
                frequencies * (self._k1 + 1.0)
                / (frequencies + self._normalizations[positions]))
        size = self._recall_size if recall_size is None else int(recall_size)
        size = max(0, size)
        if not size:
            return []
        for entity_id in excluded:
            position = self._index.get(entity_id)
            if position is not None:
                scores[position] = 0.0
        positions = np.flatnonzero(scores > 0)
        if len(positions) > size:
            candidate_scores = scores[positions]
            threshold = np.partition(candidate_scores, len(candidate_scores) - size)[
                len(candidate_scores) - size]
            positions = positions[candidate_scores >= threshold]
        ranked = sorted(positions, key=lambda position: (-scores[position], self._ids[position]))
        return [ScoreItem(item=self._ids[position], score=float(scores[position]))
                for position in ranked[:size]]
