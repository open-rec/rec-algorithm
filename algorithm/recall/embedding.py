import abc

import numpy as np
from gensim.models.word2vec import Word2Vec

from algorithm.recall.recall import EVENT_UNIQUE_COLUMNS, Recall
from algorithm.structure.score_item import ScoreItem

DEFAULT_VECTOR_SIZE = 10


class Embedding(Recall):

    def __init__(self, users=None, items=None, events=None, recall_size=1000):
        super().__init__(users=users, items=items, events=events, recall_size=recall_size)
        self._model = None

    @abc.abstractmethod
    def train(self, sentences=None, vector_size=DEFAULT_VECTOR_SIZE):
        pass

    @abc.abstractmethod
    def gen_sentences(self):
        pass

    @abc.abstractmethod
    def triggers_to_vectors(self, triggers=[]):
        pass

    @abc.abstractmethod
    def vectors_to_results(self, vectors=[]):
        pass

    @abc.abstractmethod
    def dump_vectors(self, vector_size=DEFAULT_VECTOR_SIZE):
        pass


class ItemEntityEmbedding(Embedding):
    """
    Content-based item embeddings. Not implemented — use EventEmbedding.

    These raise instead of returning None so a caller finds out immediately rather than further
    down, where an empty result looks like "no recall" rather than "no implementation".
    """

    def __init__(self, items=None, recall_size=1000):
        super().__init__(items=items, recall_size=recall_size)

    def gen_sentences(self):
        raise NotImplementedError

    def train(self, sentences=None, vector_size=DEFAULT_VECTOR_SIZE):
        raise NotImplementedError

    def triggers_to_vectors(self, triggers=[]):
        raise NotImplementedError

    def vectors_to_results(self, vectors=[]):
        raise NotImplementedError

    def recall(self, user_triggers=[], item_triggers=[]):
        raise NotImplementedError

    def dump_vectors(self, vector_size=DEFAULT_VECTOR_SIZE):
        raise NotImplementedError


class EventEmbedding(Embedding):
    """
    Behaviour-sequence embeddings: word2vec over each user's item sequence, so items appearing in
    similar contexts end up close together.
    """

    def __init__(self, events=None, recall_size=1000):
        super().__init__(events=events, recall_size=recall_size)

    def train(self, sentences=None, vector_size=DEFAULT_VECTOR_SIZE):
        if sentences:
            self._model = Word2Vec(sentences=sentences, vector_size=vector_size, min_count=5,
                                   window=5, epochs=3)
        return self._model

    def gen_sentences(self):
        # drop_duplicates and sort_values return new frames instead of mutating in place, so their
        # results have to be kept. Order matters: the word2vec window slides along each sequence, so
        # an unsorted frame trains on orderings that never occurred.
        events = self._events.drop_duplicates(subset=EVENT_UNIQUE_COLUMNS).sort_values('time')
        return events.groupby('user_id')['item_id'].apply(list).tolist()

    def _ensure_model(self, vector_size=DEFAULT_VECTOR_SIZE):
        """Train once per instance — recall() used to retrain the whole model on every call."""
        if self._model is None:
            self.train(sentences=self.gen_sentences(), vector_size=vector_size)
        return self._model

    def triggers_to_vectors(self, triggers=[]):
        """Vectors for the triggers that made it into the vocabulary; unknown ones are skipped."""
        model = self._ensure_model()
        if model is None:
            return []
        return [model.wv[trigger] for trigger in triggers if trigger in model.wv]

    def vectors_to_results(self, vectors=[]):
        """
        Nearest neighbours of the average of the given vectors — the same shape as the online
        EmbeddingNode, which averages trigger vectors before its kNN search.
        """
        model = self._ensure_model()
        if model is None or len(vectors) == 0:
            return []
        centroid = np.mean(np.asarray(vectors, dtype=float), axis=0)
        return [ScoreItem(item=item, score=score)
                for item, score in model.wv.similar_by_vector(centroid, topn=self._recall_size)]

    def recall(self, user_triggers=[], item_triggers=[]):
        triggers = item_triggers
        assert triggers

        model = self._ensure_model()
        if model is None:
            return []

        # min_count keeps rare items out of the vocabulary, so a trigger may simply not be there.
        # most_similar raises KeyError on those, which used to bring the whole recall down.
        known = [trigger for trigger in triggers if trigger in model.wv]
        if not known:
            return []

        return [ScoreItem(item=item, score=score)
                for item, score in model.wv.most_similar(positive=known, topn=self._recall_size)]

    def dump_vectors(self, vector_size=DEFAULT_VECTOR_SIZE):
        model = self._ensure_model(vector_size=vector_size)
        if model is None:
            return []
        return [(item, model.wv[item].tolist()) for item in model.wv.index_to_key]
