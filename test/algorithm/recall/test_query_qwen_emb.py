import numpy as np

from algorithm.recall.query_embedding import QueryEmbeddingRecall
from algorithm.recall.query_qwen_emb import QueryQwenEmbeddingRecall


def test_qwen_recall_uses_matching_vectors_and_excludes_seen_candidates():
    # Small synthetic catalog in the experiment's 4096-dimensional space.
    vectors = np.zeros((3, 4096), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    vectors[2, 0] = -1
    recall = QueryQwenEmbeddingRecall(['a', 'b', 'c'], vectors, recall_size=2)
    results = recall.recall(vectors[0], exclude=['a'])
    assert [row.item for row in results] == ['b', 'c']
    assert [row.score for row in results] == [0, -1]
    assert isinstance(recall, QueryEmbeddingRecall)
