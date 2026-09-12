import math
import json

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession

from algorithm.recall.hot import Hot
from algorithm.recall.content_i2i import ContentBasedI2I
from algorithm.recall.item_cf_i2i import ItemBasedI2I
from algorithm.recall.user_cf_u2i import UserBasedCF
from jobs.spark.recall import content_i2i, hot, item_cf_i2i, user_cf_u2i
from jobs.spark.io import read_items, read_users, resolve_event_history
from jobs.spark.rank_job import _freeze_feature_history, _user_pair_events
from jobs.spark.point_in_time import materialize_point_in_time_samples_spark


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[2]").appName("openrec-parity").getOrCreate()
    yield session
    session.stop()


def events():
    return pd.DataFrame([
        ("e1", "u1", "a", 1, "click", "1", "home"),
        ("e2", "u1", "b", 2, "click", "1", "home"),
        ("e3", "u2", "a", 3, "click", "1", "home"),
        ("e4", "u2", "c", 4, "click", "1", "home"),
        ("e4", "u2", "c", 4, "click", "1", "home"),
    ], columns=["id", "user_id", "item_id", "time", "type", "value", "scene"])


def test_hot_matches_local_formula(spark):
    source = events()
    local = {row.item: row.score for row in Hot(source, 10).recall()}
    distributed = {row.item: row.score for row in hot(spark.createDataFrame(source), 10).collect()}
    assert distributed == local


def test_i2i_matches_local_formula(spark):
    source = events()
    local = ItemBasedI2I(source).dump_i2i(10)
    distributed = {(row.left_item, row.right_item): row.score
                   for row in item_cf_i2i(spark.createDataFrame(source), 10).collect()}
    for left, neighbours in local.items():
        for right, score in neighbours:
            assert math.isclose(distributed[(left, right)], score, rel_tol=1e-12)


def test_user_cf_matches_local_formula(spark):
    source = events()
    local = UserBasedCF(source, recall_size=10).dump_user_recall()
    distributed = {(row.user, row.item): row.score
                   for row in user_cf_u2i(spark.createDataFrame(source), 10).collect()}
    for user, candidates in local.items():
        for item, score in candidates:
            assert math.isclose(distributed[(user, item)], score, rel_tol=1e-12)


def test_content_matches_local_formula(spark):
    source = pd.DataFrame([
        ("a", "home", "movie/action", "hero,space", "space hero"),
        ("b", "home", "movie/action", "hero", "another hero"),
        ("c", "home", "book/history", "ancient", "old world"),
    ], columns=["id", "scene", "category", "tags", "title"])
    local = ContentBasedI2I(source).dump_i2i(10)
    distributed = {(row.left_item, row.right_item): row.score
                   for row in content_i2i(spark.createDataFrame(source), 10).collect()}
    for left, neighbours in local.items():
        for right, score in neighbours:
            assert math.isclose(distributed[(left, right)], score, rel_tol=1e-12)


def test_feature_history_is_strictly_before_every_label(spark):
    source = spark.createDataFrame([
        ("past", 99), ("first-label", 100), ("future", 101),
    ], ["id", "time"])
    labels = source.filter("time >= 100")

    history, cutoff = _freeze_feature_history(source, labels)

    assert cutoff == 100
    assert [(row.id, row.time) for row in history.collect()] == [("past", 99)]


def test_user_pair_labels_are_balanced_and_keep_source_label_time(spark):
    source = spark.createDataFrame([
        ("home", "u1", "a", 10, "click"),
        ("home", "u2", "a", 11, "collect"),
        ("home", "u3", "c", 30, "click"),
    ], ["scene", "user_id", "item_id", "time", "type"])
    users = spark.createDataFrame([("u1",), ("u2",), ("u3",), ("inactive",)], ["id"])

    labels = _user_pair_events(source, users, max_events=20).collect()

    positives = [row for row in labels if row.type == "click"]
    negatives = [row for row in labels if row.type == "expose"]
    assert len(positives) == len(negatives) == 2
    assert {(row.user_id, row.item_id) for row in positives} == {("u1", "u2"), ("u2", "u1")}
    assert {row.item_id for row in negatives} == {"u3"}
    assert {row.time for row in negatives} == {11}
    assert all("inactive" not in (row.user_id, row.item_id) for row in labels)


def test_entity_snapshots_respect_second_cutoff_and_millisecond_mutations(spark, tmp_path):
    item_path = str(tmp_path / "items")
    user_path = str(tmp_path / "users")
    before_item = json.dumps({
        "schemaVersion": 1, "entityType": "item", "operation": "INSERT",
        "occurredAt": 1000000000000, "data": {"id": "i", "scene": "home", "weight": 1},
    })
    after_item = json.dumps({
        "schemaVersion": 1, "entityType": "item", "operation": "UPDATE",
        "occurredAt": 2000000000000, "data": {"id": "i", "scene": "home", "weight": 9},
    })
    before_user = json.dumps({
        "schemaVersion": 1, "entityType": "user", "operation": "INSERT",
        "occurredAt": 1000000000000, "data": {"id": "u", "city": "before"},
    })
    after_user = json.dumps({
        "schemaVersion": 1, "entityType": "user", "operation": "UPDATE",
        "occurredAt": 2000000000000, "data": {"id": "u", "city": "after"},
    })
    spark.createDataFrame([(before_item, "1970-01-01"), (after_item, "1970-01-01")],
                          ["json", "dt"]).write.partitionBy("dt").mode("overwrite").text(item_path)
    spark.createDataFrame([(before_user, "1970-01-01"), (after_user, "1970-01-01")],
                          ["json", "dt"]).write.partitionBy("dt").mode("overwrite").text(user_path)

    items = read_items(spark, date="1970-01-01", cumulative=True,
                       path=item_path, as_of_time=1500000000)
    users = read_users(spark, date="1970-01-01", cumulative=True,
                       path=user_path, as_of_time=1500000000)

    assert items.select("id", "weight").first().asDict() == {"id": "i", "weight": 1.0}
    assert users.select("id", "city").first().asDict() == {"id": "u", "city": "before"}


def test_label_population_uses_fixed_observation_cutoff_and_delete_wins_tie(spark):
    rows = [
        ("e1", "u", "i", "s", "click", "t1", 100, "INSERT", 110, "d1"),
        # Future update must not rewrite the label at observation cutoff 150.
        ("e1", "u", "i", "s", "expose", "t1", 100, "UPDATE", 200, "d2"),
        # A late-arriving old event is invisible because its mutation arrived after the cutoff.
        ("late", "u", "i", "s", "click", "t2", 90, "INSERT", 170, "d2"),
        ("tie", "u", "i", "s", "click", "t3", 120, "INSERT", 140, "d1"),
        ("tie", "u", "i", "s", "click", "t3", 120, "DELETE", 140, "d1"),
    ]
    frame = spark.createDataFrame(rows, ["event_id", "user_id", "item_id", "scene", "type",
                                          "trace_id", "time", "_operation",
                                          "_effective_time", "dt"])
    actual = resolve_event_history(frame, 150).select("event_id", "type").collect()
    assert [row.asDict() for row in actual] == [{"event_id": "e1", "type": "click"}]


def test_spark_materializes_aligned_point_in_time_features(spark):
    labels = spark.createDataFrame([
        ("l1", "u", "i", "s", "expose", 100, "x1"),
        ("l2", "u", "i", "s", "click", 200, "x2"),
    ], ["event_id", "user_id", "item_id", "scene", "type", "time", "trace_id"])
    history = spark.createDataFrame([
        ("e1", "legacy1", "u", "i", "s", "click", "1", 50, "h1", "INSERT", 60, 60, "d1"),
        ("e2", "legacy2", "u", "i", "s", "click", "1", 150, "h2", "INSERT", 160, 160, "d1"),
    ], ["event_id", "id", "user_id", "item_id", "scene", "type", "value", "time",
        "trace_id", "_operation", "_mutation_time", "_effective_time", "dt"])
    users = spark.createDataFrame([
        ("u", "old", "INSERT", 10, 10, "d1"),
        ("u", "new", "UPDATE", 150, 150, "d1"),
    ], ["id", "city", "_operation", "_mutation_time", "_effective_time", "dt"])
    items = spark.createDataFrame([
        ("i", "s", 1.0, "INSERT", 10, 10, "d1"),
    ], ["id", "scene", "weight", "_operation", "_mutation_time", "_effective_time", "dt"])

    out_labels, out_users, out_items = materialize_point_in_time_samples_spark(
        labels, history, users, items)
    assert out_labels.count() == 2
    user_rows = {row._sample_id: row for row in out_users.collect()}
    first, second = user_rows["l1"], user_rows["l2"]
    assert (first.city, first.event_count) == ("old", 1.0)
    assert (second.city, second.event_count) == ("new", 2.0)
    assert sorted(row.event_count for row in out_items.collect()) == [1.0, 2.0]
