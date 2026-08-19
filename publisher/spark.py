"""Spark partition publishers matching rec-server's Redis and Elasticsearch contracts."""

import json


def publish_redis(frame, kind, host="redis", port=6379, password=None):
    if kind not in ("hot", "new", "i2i"):
        raise ValueError("Redis publisher supports hot, new and i2i")
    key_columns = ["scene", "left_item"] if kind == "i2i" else ["scene"]
    partitioned = frame.repartition(*key_columns)

    def write(rows):
        import redis
        client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
        grouped = {}
        for row in rows:
            if kind == "i2i":
                key = "i2i:{%s}:%s" % (row.left_item, row.scene)
                member = row.right_item
            else:
                key = "%s:{%s}" % (kind, row.scene)
                member = row.item
            grouped.setdefault(key, []).append((member, float(row.score)))
        for key, values in grouped.items():
            pipe = client.pipeline(transaction=True)
            pipe.delete(key)
            if values:
                pipe.zadd(key, dict(values))
            pipe.execute()
        client.close()

    partitioned.foreachPartition(write)


def publish_embedding(frame, hosts, user=None, password=None, verify_certs=True,
                      ca_certs=None, index_suffix="item-vector-index"):
    """Replace per-scene vector indexes consumed by rec-server's EmbeddingNode."""
    scenes = [(row.scene, len(row.vector)) for row in frame.select("scene", "vector")
              .groupBy("scene").first().collect()]
    from elasticsearch import Elasticsearch
    client = Elasticsearch(hosts, basic_auth=(user, password) if user else None,
                           verify_certs=verify_certs, ca_certs=ca_certs)
    for scene, dim in scenes:
        name = "%s-%s" % (scene, index_suffix)
        if client.indices.exists(index=name):
            client.indices.delete(index=name)
        client.indices.create(index=name, mappings={"properties": {
            "id": {"type": "keyword"},
            "vector": {"type": "dense_vector", "dims": dim, "index": True,
                       "similarity": "cosine"},
        }})
    client.close()

    def write(rows):
        from elasticsearch import Elasticsearch, helpers
        es = Elasticsearch(hosts, basic_auth=(user, password) if user else None,
                           verify_certs=verify_certs, ca_certs=ca_certs)
        actions = ({"_index": "%s-%s" % (row.scene, index_suffix), "_id": row.item,
                    "_source": {"id": row.item, "vector": list(row.vector)}} for row in rows)
        helpers.bulk(es, actions)
        es.close()

    frame.repartition("scene").foreachPartition(write)
