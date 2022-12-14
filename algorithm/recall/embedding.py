import abc

from gensim.models.word2vec import Word2Vec

from algorithm.recall.recall import Recall
from algorithm.structure.score_item import ScoreItem


class Embedding(Recall):

    def __init__(self, users=None, items=None, events=None, recall_size=1000):
        super().__init__(users=users, items=items, events=events, recall_size=recall_size)
        self._model = None

    @abc.abstractmethod
    def train(self):
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
    def dump_vectors(self):
        pass


class ItemEntityEmbedding(Embedding):

    def __init__(self, items=None, recall_size=1000):
        super().__init__(items=items, recall_size=recall_size)

    def gen_sentences(self):
        pass

    def train(self):
        pass

    def triggers_to_vectors(self, triggers=[]):
        pass

    def vectors_to_results(self, vectors=[]):
        pass

    def recall(self, user_triggers=[], item_triggers=[]):
        pass

    def dump_vectors(self):
        pass


class EventEmbedding(Embedding):

    def __init__(self, events=None, recall_size=1000):
        super().__init__(events=events, recall_size=recall_size)
        self._model = None

    def train(self, sentences=None, vector_size=10):
        if sentences:
            self._model = Word2Vec(sentences=sentences, vector_size=vector_size, min_count=5, window=5, epochs=3)

    def gen_sentences(self):
        self._events.drop_duplicates((['id', 'user_id', 'item_id', 'time', 'type', 'value']))
        self._events.sort_values('time')
        user_item_sequence = self._events.groupby('user_id')[['item_id', 'time']].apply(lambda x: list(x['item_id'])) \
            .reset_index().rename(columns={0: 'item_sequence'})
        return user_item_sequence['item_sequence'].values.tolist()

    def triggers_to_vectors(self, triggers=[]):
        pass

    def vectors_to_results(self, vectors=[]):
        pass

    def recall(self, user_triggers=[], item_triggers=[]):
        triggers = item_triggers
        assert triggers

        item_sentences = self.gen_sentences()
        self.train(sentences=item_sentences)
        recall_items = []
        for item, score in self._model.wv.most_similar(positive=triggers, topn=self._recall_size):
            recall_items.append(ScoreItem(item=item, score=score))
        return recall_items

    def dump_vectors(self, vector_size=10):
        item_sentences = self.gen_sentences()
        self.train(sentences=item_sentences, vector_size=vector_size)
        return [(item, self._model.wv[item].tolist()) for item in self._model.wv.index_to_key]
