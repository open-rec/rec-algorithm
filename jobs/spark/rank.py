"""Distributed rank sample construction; model training remains PyTorch-compatible."""

from pyspark.sql import functions as F


def labelled_interactions(events, users, items):
    """Join labels to as-of entities; the item snapshot excludes latest DELETE tombstones."""
    labels = events.filter(F.col("type").isin("click", "expose")) \
        .withColumn("label", (F.col("type") == "click").cast("double"))
    user_columns = [F.col("u.%s" % name).alias("user_%s" % name)
                    for name in users.columns if name != "id"]
    item_columns = [F.col("i.%s" % name).alias("item_%s" % name)
                    for name in items.columns if name != "id"]
    return labels.alias("e").join(users.alias("u"), F.col("e.user_id") == F.col("u.id"), "inner") \
        .join(items.alias("i"), F.col("e.item_id") == F.col("i.id"), "inner") \
        .select(F.col("e.scene"), F.col("e.user_id"), F.col("e.item_id"),
                F.col("e.time"), F.col("label"), *user_columns, *item_columns)


def deterministic_split(samples, validation_percent=20):
    """Stable split independent of input partitioning, suitable for scheduled reruns."""
    bucket = F.pmod(F.xxhash64("user_id", "item_id", "time"), F.lit(100))
    return (samples.filter(bucket >= validation_percent),
            samples.filter(bucket < validation_percent))
