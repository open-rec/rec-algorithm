"""Qwen query-to-item recall over precomputed, matching Qwen embeddings."""

from algorithm.recall.query_embedding import QueryEmbeddingRecall


class QueryQwenEmbeddingRecall(QueryEmbeddingRecall):
    """Cosine recall for Qwen query/catalog vectors.

    Encoding is performed upstream. The Music CRS experiment uses frozen
    Qwen/Qwen3-Embedding-8B, last-token pooling and L2 normalization (4096D).
    Both query and candidate vectors must use the same model and encoding recipe.
    Hash, ALS and sequence embeddings do not belong to this strategy.
    """
