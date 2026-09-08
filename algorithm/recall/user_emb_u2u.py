import numpy as np


class UserEmbeddingU2U:
    """Dense user embeddings learned by truncated SVD over user-item behaviour."""

    def __init__(self, events, vector_size=32, neighbour_size=100):
        self.events = events
        self.vector_size = vector_size
        self.neighbour_size = neighbour_size
        self._vectors = None

    def dump_vectors(self):
        if self._vectors is not None:
            return self._vectors
        events = self.events.drop_duplicates(subset=['user_id', 'item_id'])
        users = sorted(events['user_id'].unique(), key=str)
        items = sorted(events['item_id'].unique(), key=str)
        user_index = {value: index for index, value in enumerate(users)}
        item_index = {value: index for index, value in enumerate(items)}
        matrix = np.zeros((len(users), len(items)), dtype=float)
        for row in events.itertuples():
            matrix[user_index[row.user_id], item_index[row.item_id]] = 1.0
        if matrix.size == 0:
            self._vectors = {}
            return self._vectors
        left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        dimensions = min(self.vector_size, len(singular))
        embeddings = left[:, :dimensions] * singular[:dimensions]
        self._vectors = {user: embeddings[index].tolist()
                         for user, index in user_index.items()}
        return self._vectors

    def dump_u2u(self):
        vectors = self.dump_vectors()
        users = list(vectors)
        if not users:
            return {}
        matrix = np.asarray([vectors[user] for user in users], dtype=float)
        norms = np.linalg.norm(matrix, axis=1)
        similarities = matrix.dot(matrix.T) / np.maximum(np.outer(norms, norms), 1e-12)
        return {left: sorted(((right, float(similarities[row, column]))
                              for column, right in enumerate(users) if row != column),
                             key=lambda value: (-value[1], str(value[0])))[:self.neighbour_size]
                for row, left in enumerate(users)}

    def vector_rows(self, scene='default'):
        return [{'scene': scene, 'id': user, 'vector': vector}
                for user, vector in self.dump_vectors().items()]

    def rows(self, scene='default'):
        return [{'scene': scene, 'left_user': left, 'right_user': right, 'score': score}
                for left, related in self.dump_u2u().items() for right, score in related]
