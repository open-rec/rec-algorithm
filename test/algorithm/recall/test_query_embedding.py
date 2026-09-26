import numpy as np
import pytest

from algorithm.recall.query_embedding import QueryEmbeddingRecall


def test_query_embedding_recall_normalizes_excludes_and_breaks_ties_by_id():
    recall = QueryEmbeddingRecall(
        ["b", "a", "c"],
        np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        recall_size=3,
    )

    result = recall.recall([2, 0], exclude=["b"])

    assert [item.item for item in result] == ["a", "c"]
    assert result[0].score == pytest.approx(1.0)


def test_query_embedding_recall_validates_shared_space_and_empty_query():
    recall = QueryEmbeddingRecall(["a"], [[1, 0]])
    assert recall.recall([0, 0]) == []
    with pytest.raises(ValueError, match="dimension"):
        recall.recall([1, 0, 0])


def test_query_embedding_recall_many_matches_single_query_semantics():
    recall = QueryEmbeddingRecall(["a", "b", "c"], [[1, 0], [0, 1], [-1, 0]], 2)
    result = recall.recall_many([[1, 0], [0, 1]], excludes=[["a"], []], block_size=1)
    assert [[item.item for item in row] for row in result] == [["b", "c"], ["b", "a"]]
