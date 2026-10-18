"""Global item-title vectors from the same OpenAI-compatible model used online."""

import itertools
import json
import math
import urllib.request


def embed_titles(texts, endpoint, model, dimensions, api_key="", timeout=60):
    """Titles are documents: send raw text, without the query instruction prefix."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(endpoint, headers=headers, data=json.dumps({
        "model": model, "input": texts, "encoding_format": "float",
    }).encode())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("model") != model:
        raise ValueError("embedding model mismatch")
    data = payload.get("data", [])
    if len(data) != len(texts):
        raise ValueError("embedding response count mismatch")
    ordered = {}
    for entry in data:
        position, vector = entry.get("index"), entry.get("embedding")
        if type(position) is not int or not 0 <= position < len(texts) or position in ordered:
            raise ValueError("invalid embedding response index")
        if not isinstance(vector, list) or len(vector) != dimensions or not all(
                type(value) in (int, float) and math.isfinite(value) for value in vector):
            raise ValueError("invalid embedding dimensions or values")
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("invalid embedding norm")
        ordered[position] = [value / norm for value in vector]
    return [ordered[position] for position in range(len(texts))]


def title_embeddings(items, endpoint, model, dimensions=1024, api_key="", batch_size=32,
                     partitions=1, timeout=60):
    from pyspark.sql import functions as F
    if dimensions <= 0 or batch_size <= 0 or partitions <= 0 or timeout <= 0:
        raise ValueError("embedding limits must be positive")
    # read_items already resolves the latest global ID and removes tombstones.
    titles = items.filter(F.col("id").isNotNull() & (F.length(F.trim("id")) > 0)
                          & F.col("title").isNotNull() & (F.length(F.trim("title")) > 0)) \
        .select(F.col("id").alias("item"), "title")
    if titles.groupBy("item").count().filter("count > 1").limit(1).count():
        raise ValueError("title snapshot must contain unique global item IDs")

    def encode(rows):
        rows = iter(rows)
        while True:
            batch = list(itertools.islice(rows, batch_size))
            if not batch:
                return
            vectors = embed_titles([row.title for row in batch], endpoint, model,
                                   dimensions, api_key, timeout)
            for row, vector in zip(batch, vectors):
                yield row.item, vector

    return items.sparkSession.createDataFrame(
        titles.repartition(partitions).rdd.mapPartitions(encode),
        "item string, vector array<double>")
