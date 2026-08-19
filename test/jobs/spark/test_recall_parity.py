import math

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession

from algorithm.recall.hot import Hot
from algorithm.recall.i2i import ItemBasedI2I
from jobs.spark.recall import hot, i2i


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
