"""Aggregate recommendation business metrics from daily Hive ODS event partitions."""

import argparse
import json
import re

from pyspark.sql import SparkSession, Window, functions as F

from jobs.spark.io import EVENT_FIELDS, read_entity


def parser():
    result = argparse.ArgumentParser(description="OpenRec business analytics job")
    result.add_argument("--date-from", required=True)
    result.add_argument("--date-to", required=True)
    result.add_argument("--scene", default="")
    result.add_argument("--event-path", default="hdfs://namenode:8020/openrec/hive/event")
    return result


def _validate(args):
    for value in (args.date_from, args.date_to):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", value or ""):
            raise ValueError("dates must use YYYY-MM-DD")
    if args.date_from > args.date_to:
        raise ValueError("date_from must not be after date_to")
    if args.scene and not re.match(r"^[A-Za-z0-9_-]+$", args.scene):
        raise ValueError("invalid scene")


def _metrics(frame):
    typed = frame.withColumn("is_expose", (F.col("type") == "expose").cast("long")) \
        .withColumn("is_click", (F.col("type") == "click").cast("long")) \
        .withColumn("is_buy", (F.col("type") == "buy").cast("long"))
    quantity = F.coalesce(
        F.get_json_object("ext_fields", "$.quantity").cast("double"),
        F.get_json_object("ext_fields", "$.count").cast("double"))
    price = F.coalesce(
        F.get_json_object("ext_fields", "$.price").cast("double"),
        F.get_json_object("ext_fields", "$.unitPrice").cast("double"))
    return typed.withColumn(
        "gmv", F.when((F.col("is_buy") == 1) & (quantity >= 0) & (price >= 0),
                      quantity * price).otherwise(F.lit(0.0))) \
        .withColumn("gmv_valid", ((F.col("is_buy") == 1) & quantity.isNotNull()
                                   & price.isNotNull()).cast("long"))


def _summary(frame):
    row = frame.agg(
        F.sum("is_expose").alias("exposes"), F.sum("is_click").alias("clicks"),
        F.sum("is_buy").alias("purchases"),
        F.countDistinct(F.when(F.col("is_expose") == 1, F.col("user_id"))).alias("browsing_users"),
        F.countDistinct(F.when(F.col("is_click") == 1, F.col("user_id"))).alias("click_users"),
        F.countDistinct(F.when(F.col("is_buy") == 1, F.col("user_id"))).alias("purchase_users"),
        F.countDistinct("item_id").alias("active_items"), F.sum("gmv").alias("gmv"),
        F.sum("gmv_valid").alias("gmv_valid_orders")).first().asDict()
    values = {key: (value or 0) for key, value in row.items()}
    values["pv_ctr"] = values["clicks"] / values["exposes"] if values["exposes"] else 0.0
    values["uv_ctr"] = values["click_users"] / values["browsing_users"] if values["browsing_users"] else 0.0
    values["pv_cvr"] = values["purchases"] / values["clicks"] if values["clicks"] else 0.0
    values["uv_cvr"] = values["purchase_users"] / values["click_users"] if values["click_users"] else 0.0
    return values


def run(args, spark=None):
    _validate(args)
    spark = spark or SparkSession.builder.appName("openrec-business-analytics").getOrCreate()
    events = read_entity(spark, "openrec.event_entity", EVENT_FIELDS,
                         keep_partition=True, path=args.event_path) \
        .filter((F.col("dt") >= args.date_from) & (F.col("dt") <= args.date_to))
    if args.scene:
        events = events.filter(F.col("scene") == args.scene)
    fallback = F.sha2(F.concat_ws("|", "user_id", "item_id", "scene", "type",
                                  F.col("time").cast("string")), 256)
    keyed = events.withColumn("_event_key", F.coalesce("trace_id", "id", fallback))
    window = Window.partitionBy("_event_key").orderBy(
        F.desc("_mutation_time"), F.desc("time"), F.desc("dt"))
    metrics = _metrics(keyed.withColumn("_row", F.row_number().over(window))
                       .filter("_row = 1 AND _operation <> 'DELETE'")
                       .drop("_row", "_event_key"))
    summary = _summary(metrics)
    daily = []
    for row in metrics.groupBy("dt").agg(
            F.sum("is_expose").alias("exposes"), F.sum("is_click").alias("clicks"),
            F.sum("is_buy").alias("purchases"), F.countDistinct("item_id").alias("active_items"),
            F.sum("gmv").alias("gmv")).orderBy("dt").collect():
        value = {key: (item or 0) for key, item in row.asDict().items()}
        value["dt"] = str(value["dt"])
        value["pv_ctr"] = value["clicks"] / value["exposes"] if value["exposes"] else 0.0
        value["pv_cvr"] = value["purchases"] / value["clicks"] if value["clicks"] else 0.0
        daily.append(value)
    result = {"date_from": args.date_from, "date_to": args.date_to,
              "scene": args.scene or None, "summary": summary, "daily": daily,
              "gmv_schema": "buy.extFields.quantity * buy.extFields.price"}
    print("OPENREC_ANALYTICS=" + json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    run(parser().parse_args())
