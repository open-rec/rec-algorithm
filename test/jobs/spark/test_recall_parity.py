import math
import json

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession

from algorithm.recall.hot import Hot
from algorithm.recall.i2i import ItemBasedI2I
from jobs.spark.recall import hot, i2i
from jobs.spark.io import read_items, read_users
from jobs.spark.rank_job import _freeze_feature_history


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
                   for row in i2i(spark.createDataFrame(source), 10).collect()}
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
