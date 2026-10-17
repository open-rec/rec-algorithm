"""Local implicit-feedback ALS for the default user recall bundle."""

import numpy as np

from algorithm.recall.user_emb_u2u import UserEmbeddingU2U


class UserALSRecall(UserEmbeddingU2U):
    """Fit binary preferences using alternating regularized least squares.

    Observed pairs have preference 1 and confidence 2; unobserved pairs have
    preference 0 and confidence 1. Regularization scales with observed degree,
    as in the Spark implicit ALS job (alpha=1, regParam=0.1). Local and Spark
    random initialization differ, so their factors are not interchangeable.
    """

    def __init__(self, events, vector_size=32, neighbour_size=100, max_iter=5,
                 regularization=0.1, seed=7):
        super().__init__(events, vector_size, neighbour_size)
        if vector_size < 1 or max_iter < 1 or regularization <= 0:
            raise ValueError("ALS dimensions, iterations and regularization must be positive")
        self.max_iter = max_iter
        self.regularization = regularization
        self.seed = seed

    def _solve(self, factors, neighbours):
        gram = factors.T @ factors
        identity = np.eye(self.vector_size)
        result = np.empty((len(neighbours), self.vector_size))
        for index, observed in enumerate(neighbours):
            selected = factors[observed]
            matrix = gram + selected.T @ selected
            matrix += self.regularization * max(len(observed), 1) * identity
            result[index] = np.linalg.solve(matrix, 2 * selected.sum(axis=0))
        return result

    def dump_vectors(self):
        if self._vectors is not None:
            return self._vectors
        events = self.events.dropna(subset=['user_id', 'item_id']).drop_duplicates(
            subset=['user_id', 'item_id'])
        users = sorted(events['user_id'].unique(), key=str)
        items = sorted(events['item_id'].unique(), key=str)
        user_index = {value: index for index, value in enumerate(users)}
        item_index = {value: index for index, value in enumerate(items)}
        user_items, item_users = [[] for _ in users], [[] for _ in items]
        for row in events.itertuples():
            user, item = user_index[row.user_id], item_index[row.item_id]
            user_items[user].append(item)
            item_users[item].append(user)
        # Stable adjacency order also makes results independent of CSV row ordering.
        user_items = [sorted(values) for values in user_items]
        item_users = [sorted(values) for values in item_users]
        if not users:
            self._vectors = {}
            return self._vectors
        rng = np.random.default_rng(self.seed)
        item_factors = rng.normal(0, 0.1, (len(items), self.vector_size))
        for _ in range(self.max_iter):
            user_factors = self._solve(item_factors, user_items)
            item_factors = self._solve(user_factors, item_users)
        self._vectors = {user: user_factors[index].tolist() for index, user in enumerate(users)}
        return self._vectors

    def dump_u2u(self):
        vectors = self.dump_vectors()
        users = list(vectors)
        if not users:
            return {}
        matrix = np.asarray(list(vectors.values()))
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        result = {}
        for start in range(0, len(users), 128):
            scores = matrix[start:start + 128] @ matrix.T
            for offset, row in enumerate(scores):
                index = start + offset
                row[index] = -np.inf
                order = np.argsort(-row, kind='stable')[:min(self.neighbour_size, len(users) - 1)]
                result[users[index]] = [(users[column], float(row[column])) for column in order]
        return result
