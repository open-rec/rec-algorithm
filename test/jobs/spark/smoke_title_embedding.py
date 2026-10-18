"""Opt-in real Spark/vLLM/ES check; all releases use a unique test alias/prefix."""
import os
import tempfile
import uuid
from unittest.mock import patch

from elasticsearch import Elasticsearch
from pyspark.sql import SparkSession
from jobs.spark.recall.title_embedding import title_embeddings
from jobs.spark.recall.job import parser, run
from publisher.spark import publish_recall
from rec_console.module.recall.service import RecallIndexManager


def main():
    token = "openrec-title-it-" + uuid.uuid4().hex

    class IsolatedManager(RecallIndexManager):
        def alias_name(self, algorithm):
            self._algorithm(algorithm)
            return token + "-active"

    es = Elasticsearch(["https://elasticsearch:9200"],
                       basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD", "openrec-es-password")),
                       verify_certs=False)
    spark = SparkSession.builder.master("local[2]").appName(token).getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "2")
    try:
        with tempfile.TemporaryDirectory() as directory:
            manager = IsolatedManager(es, prefix=token, data_root=directory)

            def control(url, path, payload):
                if path.endswith("prepare"):
                    return manager.prepare(**payload)
                return manager.activate(payload["algorithm"], payload["index"],
                                        payload["expected_documents"], payload["max_index_versions"])

            items = spark.createDataFrame([
                ("one", "Blue Train", "scene-a"), ("two", "Kind of Blue", "scene-b"),
                ("blank", "", "scene-a")], ["id", "title", "scene"])
            vectors = title_embeddings(items, "http://vllm:8000/v1/embeddings",
                                      "Qwen/Qwen3-Embedding-0.6B", batch_size=2).cache()
            assert vectors.count() == 2
            assert all(len(row.vector) == 1024 for row in vectors.collect())
            with patch("publisher.spark._console_request", side_effect=control):
                first = publish_recall(vectors, "query-title-emb", "2026-10-17", "r001",
                                       ["https://elasticsearch:9200"], "elastic",
                                       os.getenv("ELASTIC_PASSWORD", "openrec-es-password"), verify_certs=False)
                second = publish_recall(vectors, "query-title-emb", "2026-10-17", "r002",
                                        ["https://elasticsearch:9200"], "elastic",
                                        os.getenv("ELASTIC_PASSWORD", "openrec-es-password"), verify_certs=False)
                assert second["previous_indexes"] == [first["index"]]
                assert manager.rollback("query-title-emb")["index"] == first["index"]
                assert es.count(index=token + "-active")["count"] == 2
            duplicate = spark.createDataFrame([("one", "A"), ("one", "B")], ["id", "title"])
            try:
                title_embeddings(duplicate, "http://vllm:8000/v1/embeddings", "unused")
                raise AssertionError("duplicate global IDs accepted")
            except ValueError:
                pass
            with patch("jobs.spark.recall.inputs.read_items", return_value=items), patch(
                    "publisher.spark._console_request", side_effect=control):
                args = parser().parse_args(["query_title_emb", "--date", "2026-10-17",
                    "--revision", "r003", "--publish", "--es-password",
                    os.getenv("ELASTIC_PASSWORD", "openrec-es-password"),
                    "--output-path", directory + "/vectors"])
                run(args, spark)
                persisted = spark.read.parquet(directory + "/vectors")
                assert set(persisted.columns) == {"item", "vector", "dt"}
                assert persisted.count() == 2
                assert manager.list_indexes("query-title-emb")["active_indexes"] == [
                    token + "-query-title-emb-20261017-r003"]
            print("TITLE_PIPELINE_OK: real Spark -> vLLM -> ES; global IDs, activation and rollback")
    finally:
        for index in es.indices.get(index=token + "-*", allow_no_indices=True, ignore_unavailable=True):
            es.indices.delete(index=index)
        spark.stop()
        es.close()


if __name__ == "__main__":
    main()
