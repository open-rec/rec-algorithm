"""Request-query to entity retrieval over a shared dense embedding space."""

import numpy as np

from algorithm.structure.score_item import ScoreItem


class QueryEmbeddingRecall(object):
    """Exact cosine retrieval with the same semantics as rec-server ANN recall.

    The class is intentionally independent of any text encoder. Callers may use
    any model as long as query and entity vectors share a space.
    """

    def __init__(self, entity_ids, entity_vectors, recall_size=100, device="cpu"):
        ids = [str(value) for value in entity_ids]
        vectors = np.asarray(entity_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
            raise ValueError("entity vectors must be a nonempty matrix aligned with entity ids")
        if len(ids) != len(set(ids)):
            raise ValueError("entity ids must be unique")
        if not np.isfinite(vectors).all():
            raise ValueError("entity vectors must be finite")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._ids = np.asarray(ids, dtype=object)
        self._vectors = vectors / np.maximum(norms, 1e-12)
        self._recall_size = int(recall_size)
        self._device = str(device)
        self._torch_vectors = None
        if self._device != "cpu":
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("PyTorch is required for GPU query embedding recall") from error
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for query embedding recall")
            self._torch_vectors = torch.from_numpy(
                np.ascontiguousarray(self._vectors.T)).to(self._device)

    def recall(self, query_vector, exclude=None, recall_size=None):
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._vectors.shape[1]:
            raise ValueError("query vector dimension does not match entity vectors")
        if not np.isfinite(query).all() or np.linalg.norm(query) <= 1e-12:
            return []
        scores = self._vectors @ (query / np.linalg.norm(query))
        excluded = {str(value) for value in (exclude or ())}
        available = np.asarray([value not in excluded for value in self._ids])
        indices = np.flatnonzero(available)
        size = min(max(0, self._recall_size if recall_size is None else int(recall_size)), len(indices))
        if size == 0:
            return []
        selected = self._top(indices, scores, size)
        return [ScoreItem(item=self._ids[position], score=float(scores[position]))
                for position in selected]

    def recall_many(self, query_vectors, excludes=None, recall_size=None, block_size=128):
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self._vectors.shape[1]:
            raise ValueError("query vector matrix dimension does not match entity vectors")
        excludes = excludes or [None] * len(queries)
        if len(excludes) != len(queries):
            raise ValueError("excludes must align with query vectors")
        size = max(0, self._recall_size if recall_size is None else int(recall_size))
        id_to_position = {value: position for position, value in enumerate(self._ids)}
        results = []
        for start in range(0, len(queries), block_size):
            block = queries[start:start + block_size]
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            valid = np.isfinite(block).all(axis=1) & (norms[:, 0] > 1e-12)
            normalized = block / np.maximum(norms, 1e-12)
            if self._torch_vectors is None:
                scores = normalized @ self._vectors.T
            else:
                import torch
                with torch.no_grad():
                    scores = (torch.from_numpy(np.ascontiguousarray(normalized))
                              .to(self._device) @ self._torch_vectors).cpu().numpy()
            for offset, row in enumerate(scores):
                if not valid[offset]:
                    results.append([])
                    continue
                for value in excludes[start + offset] or ():
                    position = id_to_position.get(str(value))
                    if position is not None:
                        row[position] = -np.inf
                count = min(size, int(np.isfinite(row).sum()))
                if count == 0:
                    results.append([])
                    continue
                selected = self._top(np.flatnonzero(np.isfinite(row)), row, count)
                results.append([ScoreItem(item=self._ids[position], score=float(row[position]))
                                for position in selected])
        return results

    def _top(self, indices, scores, size):
        """Top scores with stable entity-id ordering at the selection boundary."""
        values = scores[indices]
        partition = np.argpartition(-values, size - 1)[:size]
        threshold = float(values[partition].min())
        higher = indices[values > threshold].tolist()
        tied = indices[values == threshold].tolist()
        tied.sort(key=lambda position: self._ids[position])
        selected = higher + tied[:size - len(higher)]
        return sorted(selected,
                      key=lambda position: (-float(scores[position]), self._ids[position]))
