"""Distributed recall formulas kept in parity with algorithm.recall."""

from pyspark.ml.feature import Word2Vec
from pyspark.sql import Window
from pyspark.sql import functions as F

from algorithm.recall.recall import EVENT_UNIQUE_COLUMNS


def _events(events, event_type="click"):
    columns = [column for column in EVENT_UNIQUE_COLUMNS if column in events.columns]
    if "scene" in events.columns and "scene" not in columns:
        columns.append("scene")
    frame = events.dropDuplicates(columns).filter(
        F.col("scene").isNotNull() & F.col("item_id").isNotNull())
    return frame.filter(F.col("type") == event_type) if event_type else frame


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


def i2i(events, cut_size=20, event_type="click"):
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


def embedding(events, vector_size=10, min_count=5, window_size=5, max_iter=3,
              event_type="click"):
    """Train one distributed Word2Vec model per scene; scene cardinality should stay bounded."""
    frame = _events(events, event_type).filter(F.col("user_id").isNotNull())
    scenes = [row.scene for row in frame.select("scene").distinct().collect()]
    results = []
    for scene in scenes:
        sentences = frame.filter(F.col("scene") == scene).groupBy("user_id").agg(
            F.sort_array(F.collect_list(F.struct("time", "item_id"))).alias("events")) \
            .select(F.expr("transform(events, x -> x.item_id)").alias("sentence"))
        model = Word2Vec(vectorSize=vector_size, minCount=min_count, windowSize=window_size,
                         maxIter=max_iter, inputCol="sentence", outputCol="vector").fit(sentences)
        results.append(model.getVectors().select(F.lit(scene).alias("scene"),
                                                  F.col("word").alias("item"), "vector"))
    if not results:
        return events.sparkSession.createDataFrame([], "scene string, item string, vector array<float>")
    result = results[0]
    for extra in results[1:]:
        result = result.unionByName(extra)
    return result
