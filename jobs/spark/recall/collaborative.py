"""Distributed collaborative recall formulas."""

from pyspark.sql import Window, functions as F
from .events import _events


def item_cf_i2i(events, cut_size=20, event_type="click"):
    frame = _events(events, event_type).filter(F.col("user_id").isNotNull())
    sequences = frame.groupBy("scene", "user_id").agg(
        F.sort_array(F.collect_list(F.struct("time", "item_id"))).alias("events")) \
        .select("scene", "user_id", F.expr("transform(events, x -> x.item_id)").alias("items")) \
        .withColumn("sequence_weight", 1.0 / F.log(F.size("items") + F.lit(1.0)))
    occurrences = sequences.select("scene", F.explode("items").alias("item"))
    counts = occurrences.groupBy("scene", "item").count()
    pairs = sequences.select("scene", "sequence_weight", "items",
                             F.posexplode("items").alias("li", "left_item")) \
        .select("scene", "sequence_weight", "li", "left_item",
                F.posexplode("items").alias("ri", "right_item")) \
        .filter((F.col("li") != F.col("ri")) & (F.col("left_item") != F.col("right_item"))) \
        .groupBy("scene", "left_item", "right_item").agg(F.sum("sequence_weight").alias("cooccurrence"))
    left = counts.select("scene", F.col("item").alias("left_item"), F.col("count").alias("left_count"))
    right = counts.select("scene", F.col("item").alias("right_item"), F.col("count").alias("right_count"))
    scored = pairs.join(left, ["scene", "left_item"]).join(right, ["scene", "right_item"]) \
        .withColumn("score", F.col("cooccurrence") / F.sqrt(F.col("left_count") * F.col("right_count")))
    ranked = Window.partitionBy("scene", "left_item").orderBy(F.desc("score"), F.asc("right_item"))
    return scored.withColumn("rank", F.row_number().over(ranked)).filter(F.col("rank") <= cut_size) \
        .select("scene", "left_item", "right_item", "score")


def user_cf_u2i(events, size=1000, neighbour_size=50, event_type="click"):
    """UserCF candidates with inverse-popularity weighted cosine user similarity."""
    frame = _events(events, event_type).filter(
        F.col("user_id").isNotNull() & F.col("item_id").isNotNull()) \
        .select("scene", "user_id", "item_id").dropDuplicates()
    item_users = frame.groupBy("scene", "item_id").agg(
        F.collect_set("user_id").alias("users")) \
        .withColumn("weight", 1.0 / F.log(F.size("users") + F.lit(1.0)))
    pairs = item_users.select("scene", "weight", F.explode("users").alias("user"), "users") \
        .select("scene", "weight", "user", F.explode("users").alias("neighbour")) \
        .filter(F.col("user") != F.col("neighbour")) \
        .groupBy("scene", "user", "neighbour").agg(F.sum("weight").alias("cooccurrence"))
    counts = frame.groupBy("scene", "user_id").count()
    left = counts.select("scene", F.col("user_id").alias("user"),
                         F.col("count").alias("user_count"))
    right = counts.select("scene", F.col("user_id").alias("neighbour"),
                          F.col("count").alias("neighbour_count"))
    scored = pairs.join(left, ["scene", "user"]).join(right, ["scene", "neighbour"]) \
        .withColumn("similarity", F.col("cooccurrence") /
                    F.sqrt(F.col("user_count") * F.col("neighbour_count")))
    neighbour_rank = Window.partitionBy("scene", "user").orderBy(
        F.desc("similarity"), F.asc("neighbour"))
    neighbours = scored.withColumn("rank", F.row_number().over(neighbour_rank)) \
        .filter(F.col("rank") <= neighbour_size)
    neighbour_items = frame.select("scene", F.col("user_id").alias("neighbour"),
                                   F.col("item_id").alias("item"))
    seen = frame.select("scene", F.col("user_id").alias("user"),
                        F.col("item_id").alias("item")).withColumn("seen", F.lit(1))
    candidates = neighbours.join(neighbour_items, ["scene", "neighbour"]) \
        .join(seen, ["scene", "user", "item"], "left_anti") \
        .groupBy("scene", "user", "item").agg(F.sum("similarity").alias("score"))
    candidate_rank = Window.partitionBy("scene", "user").orderBy(F.desc("score"), F.asc("item"))
    return candidates.withColumn("rank", F.row_number().over(candidate_rank)) \
        .filter(F.col("rank") <= size).select("scene", "user", "item", "score")


def user_cf_u2u(events, size=100, event_type="click"):
    """Inverse-popularity weighted cosine similarity between users."""
    frame = _events(events, event_type).filter(
        F.col("user_id").isNotNull() & F.col("item_id").isNotNull()) \
        .select("scene", "user_id", "item_id").dropDuplicates()
    item_users = frame.groupBy("scene", "item_id").agg(F.collect_set("user_id").alias("users")) \
        .withColumn("weight", 1.0 / F.log(F.size("users") + F.lit(1.0)))
    pairs = item_users.select("scene", "weight", F.explode("users").alias("left_user"), "users") \
        .select("scene", "weight", "left_user", F.explode("users").alias("right_user")) \
        .filter(F.col("left_user") != F.col("right_user")) \
        .groupBy("scene", "left_user", "right_user").agg(F.sum("weight").alias("dot"))
    counts = frame.groupBy("scene", "user_id").count()
    left = counts.select("scene", F.col("user_id").alias("left_user"),
                         F.col("count").alias("left_count"))
    right = counts.select("scene", F.col("user_id").alias("right_user"),
                          F.col("count").alias("right_count"))
    scored = pairs.join(left, ["scene", "left_user"]).join(right, ["scene", "right_user"]) \
        .withColumn("score", F.col("dot") / F.sqrt(F.col("left_count") * F.col("right_count")))
    ranked = Window.partitionBy("scene", "left_user").orderBy(F.desc("score"), F.asc("right_user"))
    return scored.withColumn("rank", F.row_number().over(ranked)).filter(F.col("rank") <= size) \
        .select("scene", "left_user", "right_user", "score")
