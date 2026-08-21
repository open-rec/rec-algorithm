"""Runnable with spark-submit; validates formulas without requiring pytest."""

import math
import json
import tempfile

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from algorithm.recall.hot import Hot
from algorithm.recall.i2i import ItemBasedI2I
from algorithm.recall.new import New
from jobs.spark.recall import embedding, hot, i2i, new
from jobs.spark.io import keep_active_item_events, read_events, read_items, write_result


def main():
    metastore = tempfile.mkdtemp(prefix="openrec-metastore-")
    warehouse = tempfile.mkdtemp(prefix="openrec-warehouse-")
    spark = SparkSession.builder.master("local[2]").appName("openrec-recall-smoke") \
        .config("javax.jdo.option.ConnectionURL",
                "jdbc:derby:;databaseName=%s/metastore_db;create=true" % metastore) \
        .config("spark.sql.warehouse.dir", warehouse) \
        .enableHiveSupport().getOrCreate()
    rows = [
        ("e1", "u1", "a", 1, "click", "1", "home"),
        ("e2", "u1", "b", 2, "click", "1", "home"),
        ("e3", "u2", "a", 3, "click", "1", "home"),
        ("e4", "u2", "c", 4, "click", "1", "home"),
        ("e4", "u2", "c", 4, "click", "1", "home"),
    ]
    columns = ["id", "user_id", "item_id", "time", "type", "value", "scene"]
    source = pd.DataFrame(rows, columns=columns)
    distributed_source = spark.createDataFrame(rows, columns)
    expected_hot = {row.item: row.score for row in Hot(source, 10).recall()}
    actual_hot = {row.item: row.score for row in hot(distributed_source, 10).collect()}
    assert actual_hot == expected_hot, (actual_hot, expected_hot)
    expected_i2i = ItemBasedI2I(source).dump_i2i(10)
    actual_i2i = {(row.left_item, row.right_item): row.score
                  for row in i2i(distributed_source, 10).collect()}
    for left, neighbours in expected_i2i.items():
        for right, score in neighbours:
            assert math.isclose(actual_i2i[(left, right)], score, rel_tol=1e-12)
    item_rows = [("a", "home", 10), ("b", "home", 20), ("c", "home", 15)]
    item_columns = ["id", "scene", "pub_time"]
    local_items = pd.DataFrame(item_rows, columns=item_columns)
    expected_new = {row.item: row.score for row in New(local_items, 10).recall()}
    actual_new = {row.item: row.score
                  for row in new(spark.createDataFrame(item_rows, item_columns), 10).collect()}
    assert actual_new == expected_new, (actual_new, expected_new)
    vectors = embedding(distributed_source, vector_size=4, min_count=1, max_iter=1).collect()
    assert vectors and all(len(row.vector) == 4 and row.scene == "home" for row in vectors)
    root = tempfile.mkdtemp(prefix="openrec-hive-day-")
    source_path = root + "/event"
    output_path = root + "/hot"
    daily_json = json.dumps({"id": "daily-1", "userId": "u1", "itemId": "a",
                             "time": 1700000000, "type": "click", "value": "1", "scene": "home"})
    previous_version = json.dumps({"id": "daily-1", "userId": "u1", "itemId": "b",
                                   "time": 1699900000, "type": "click", "value": "1", "scene": "home"})
    historical = json.dumps({"id": "historical-2", "userId": "u2", "itemId": "c",
                             "time": 1699900001, "type": "click", "value": "1", "scene": "home"})
    spark.createDataFrame([(previous_version, "2023-11-13"),
                           (historical, "2023-11-13"), (daily_json, "2023-11-14")],
                          ["json", "dt"]).write \
        .partitionBy("dt").mode("overwrite").text(source_path)
    spark.sql("CREATE DATABASE IF NOT EXISTS openrec_smoke")
    spark.sql("DROP TABLE IF EXISTS openrec_smoke.event_entity")
    spark.sql("CREATE EXTERNAL TABLE openrec_smoke.event_entity (json STRING) "
              "PARTITIONED BY (dt STRING) STORED AS TEXTFILE LOCATION '%s'" % source_path)
    daily = read_events(spark, "openrec_smoke.event_entity", "2023-11-14")
    assert daily.count() == 1 and daily.first().item_id == "a"
    cumulative = read_events(spark, "openrec_smoke.event_entity", "2023-11-14", cumulative=True)
    assert cumulative.count() == 2, cumulative.collect()
    assert {row.item_id for row in cumulative.collect()} == {"a", "c"}

    item_source_path = root + "/item"
    item_a = json.dumps({"schemaVersion": 1, "entityType": "item", "operation": "INSERT",
                         "occurredAt": 1699900000000,
                         "data": {"id": "a", "scene": "home", "pubTime": 1699900000}})
    item_c = json.dumps({"schemaVersion": 1, "entityType": "item", "operation": "INSERT",
                         "occurredAt": 1699900001000,
                         "data": {"id": "c", "scene": "home", "pubTime": 1699900001}})
    delete_c = json.dumps({"schemaVersion": 1, "entityType": "item", "operation": "DELETE",
                           "occurredAt": 1700000000000, "data": {"id": "c"}})
    spark.createDataFrame([(item_a, "2023-11-13"), (item_c, "2023-11-13"),
                           (delete_c, "2023-11-14")], ["json", "dt"]).write \
        .partitionBy("dt").mode("overwrite").text(item_source_path)
    active_items = read_items(spark, date="2023-11-14", cumulative=True,
                              path=item_source_path)
    assert [row.id for row in active_items.select("id").collect()] == ["a"]
    eligible_events = keep_active_item_events(cumulative, active_items)
    assert [row.item_id for row in eligible_events.select("item_id").collect()] == ["a"]
    write_result(hot(daily, 10).withColumn("dt", F.lit("2023-11-14")),
                 path=output_path)
    assert spark.read.parquet(output_path).filter("dt = '2023-11-14'").count() == 1
    spark.sql("DROP TABLE openrec_smoke.event_entity")
    spark.stop()
    print("Spark/local recall parity: OK")


if __name__ == "__main__":
    main()
