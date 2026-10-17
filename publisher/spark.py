"""Spark partition publishers matching rec-server's Redis and Elasticsearch contracts."""

import json
import urllib.error
import urllib.request


def _console_request(console_url, path, payload):
    request = urllib.request.Request(
        console_url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError("rec-console returned HTTP %s: %s" % (error.code, detail)) from error


def publish_recall(frame, kind, business_date, revision="r001", hosts=None, user=None,
                   password=None, verify_certs=True, ca_certs=None,
                   console_url="http://rec-console:8095", max_index_versions=2):
    """Write the staging index prepared by rec-console, then ask it to activate the release."""
    hosts = hosts or ["https://elasticsearch:9200"]
    expected_count = frame.count()
    if expected_count <= 0:
        raise ValueError("refusing to publish an empty %s recall table" % kind)
    release = _console_request(console_url, "/api/recall/releases/prepare", {
        "algorithm": kind, "business_date": business_date, "revision": revision,
    })
    index = release["index"]

    def write(rows):
        from elasticsearch import Elasticsearch, helpers
        es = Elasticsearch(hosts, basic_auth=(user, password) if user else None,
                           verify_certs=verify_certs, ca_certs=ca_certs)

        def actions():
            for row in rows:
                if kind == "query-title-emb":
                    yield {"_index": index, "_id": row.item,
                           "_source": {"id": row.item, "vector": list(row.vector)}}
                    continue
                if kind in ("item-search", "user-search"):
                    entity = "user" if kind == "user-search" else "item"
                    value = getattr(row, entity)
                    source = {"scene": row.scene, entity: value, "text": row.text}
                    doc_id = "%s:%s" % (row.scene, value)
                else:
                    source = {"scene": row.scene, "score": float(row.score)}
                if kind in ("item-cf-i2i", "content-i2i"):
                    source.update(left_item=row.left_item, right_item=row.right_item)
                    doc_id = "%s:%s:%s" % (row.scene, row.left_item, row.right_item)
                elif kind == "user-cf-u2i":
                    source.update(user=row.user, item=row.item)
                    doc_id = "%s:%s:%s" % (row.scene, row.user, row.item)
                elif kind in ("user-cf-u2u", "content-u2u", "user-als-u2u"):
                    source.update(left_user=row.left_user, right_user=row.right_user)
                    doc_id = "%s:%s:%s" % (row.scene, row.left_user, row.right_user)
                elif kind not in ("item-search", "user-search"):
                    source["item"] = row.item
                    doc_id = "%s:%s" % (row.scene, row.item)
                    if kind == "new":
                        source["publish_time"] = int(row.publish_time)
                yield {"_index": index, "_id": doc_id, "_source": source}

        helpers.bulk(es, actions())
        es.close()

    if release.get("writable", True):
        frame.repartition("item" if kind == "query-title-emb" else "scene").foreachPartition(write)
    return _console_request(console_url, "/api/recall/releases/activate", {
        "algorithm": kind,
        "index": index,
        "expected_documents": expected_count,
        "max_index_versions": max_index_versions,
    })


def publish_embedding(frame, hosts, user=None, password=None, verify_certs=True,
                      ca_certs=None, index_suffix="item-seq-emb-index",
                      business_date=None, revision="r001", max_index_versions=2):
    """Publish versioned per-scene vectors and atomically switch serving aliases."""
    version = (business_date or "legacy").replace("-", "")
    from pyspark.sql import functions as F
    scenes = [(row.scene, len(row.vector), row.documents)
              for row in frame.select("scene", "vector").groupBy("scene").agg(
                  F.first("vector").alias("vector"),
                  F.count(F.lit(1)).alias("documents")).collect()]
    if not scenes:
        raise ValueError("refusing to publish an empty embedding table")
    from elasticsearch import Elasticsearch
    client = Elasticsearch(hosts, basic_auth=(user, password) if user else None,
                           verify_certs=verify_certs, ca_certs=ca_certs)
    indexes = {}
    for scene, dim, _ in scenes:
        alias = "%s-%s" % (scene, index_suffix)
        name = "%s-%s-%s" % (alias, version, revision)
        indexes[scene] = name
        if not client.indices.exists(index=name):
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
        actions = ({"_index": indexes[row.scene], "_id": row.item,
                    "_source": {"id": row.item, "vector": list(row.vector)}} for row in rows)
        helpers.bulk(es, actions)
        es.close()

    frame.repartition("scene").foreachPartition(write)

    client = Elasticsearch(hosts, basic_auth=(user, password) if user else None,
                           verify_certs=verify_certs, ca_certs=ca_certs)
    for scene, _, expected in scenes:
        alias = "%s-%s" % (scene, index_suffix)
        index = indexes[scene]
        client.indices.refresh(index=index)
        actual = int(client.count(index=index)["count"])
        if actual != expected:
            client.close()
            raise RuntimeError("embedding index %s has %d documents, expected %d" %
                               (index, actual, expected))
        old = []
        if client.indices.exists_alias(name=alias):
            old = list(client.indices.get_alias(name=alias).keys())
        actions = [{"remove": {"index": name, "alias": alias}} for name in old]
        # Migrate the legacy concrete serving index to an alias in the same
        # atomic cluster-state update, so rec-server never observes a gap.
        if not old and client.indices.exists(index=alias):
            actions.append({"remove_index": {"index": alias}})
        actions.append({"add": {"index": index, "alias": alias}})
        client.indices.update_aliases(actions=actions)
        matching = sorted(client.indices.get(index=alias + "-*").keys(), reverse=True)
        for obsolete in matching[max_index_versions:]:
            if obsolete != index:
                client.indices.delete(index=obsolete)
    client.close()
