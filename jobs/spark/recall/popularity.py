"""Distributed popularity recall formulas."""

from pyspark.sql import Window, functions as F
from .events import _events


def hot(events, size=1000, event_type="click"):
    counts = _events(events, event_type).groupBy("scene", "item_id").count()
    scene = Window.partitionBy("scene")
    ranked = Window.partitionBy("scene").orderBy(F.desc("count"), F.asc("item_id"))
    return counts.withColumn("score", F.col("count") / F.max("count").over(scene)) \
        .withColumn("rank", F.row_number().over(ranked)).filter(F.col("rank") <= size) \
        .select("scene", F.col("item_id").alias("item"), "score")


def new(items, size=1000, power=31):
    frame = items.filter(F.col("scene").isNotNull() & F.col("id").isNotNull()) \
        .dropDuplicates(["scene", "id"]).withColumn("pub_time", F.col("pub_time").cast("double"))
    scene = Window.partitionBy("scene")
    ranked = Window.partitionBy("scene").orderBy(F.desc("pub_time"), F.asc("id"))
    frame = frame.withColumn("oldest", F.min("pub_time").over(scene)) \
        .withColumn("newest", F.max("pub_time").over(scene))
    freshness = F.when(F.col("newest") <= F.col("oldest"), F.lit(1.0)).otherwise(
        (F.col("pub_time") - F.col("oldest")) / (F.col("newest") - F.col("oldest")))
    return frame.withColumn("score", F.pow(freshness, power)) \
        .withColumn("rank", F.row_number().over(ranked)).filter(F.col("rank") <= size) \
        .select("scene", F.col("id").alias("item"), "score",
                F.col("pub_time").cast("long").alias("publish_time"))
