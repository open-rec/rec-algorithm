"""Distributed embedding recall formulas."""

from pyspark.sql import Window, functions as F
from .events import _events
from pyspark.ml.feature import Word2Vec
from pyspark.ml.recommendation import ALS


def user_als_u2u(events, size=100, vector_size=32, max_iter=5, event_type="click"):
    """ALS user factors followed by cosine U2U nearest-neighbour generation."""
    frame = _events(events, event_type).filter(
        F.col("user_id").isNotNull() & F.col("item_id").isNotNull()).select(
            "scene", "user_id", "item_id").dropDuplicates()
    scenes = [row.scene for row in frame.select("scene").distinct().collect()]
    results = []
    for scene in scenes:
        values = frame.filter(F.col("scene") == scene)
        users = values.select("user_id").distinct().withColumn(
            "user", F.row_number().over(Window.orderBy("user_id")) - 1)
        items = values.select("item_id").distinct().withColumn(
            "item", F.row_number().over(Window.orderBy("item_id")) - 1)
        ratings = values.join(users, "user_id").join(items, "item_id").withColumn("rating", F.lit(1.0))
        model = ALS(rank=vector_size, maxIter=max_iter, implicitPrefs=True, seed=7,
                    userCol="user", itemCol="item", ratingCol="rating", coldStartStrategy="drop").fit(ratings)
        factors = model.userFactors.join(users, model.userFactors.id == users.user) \
            .select("user_id", F.col("features").alias("vector")) \
            .withColumn("norm", F.sqrt(F.aggregate("vector", F.lit(0.0),
                lambda total, value: total + value * value)))
        left = factors.select(F.col("user_id").alias("left_user"),
                              F.col("vector").alias("lv"), F.col("norm").alias("ln"))
        right = factors.select(F.col("user_id").alias("right_user"),
                               F.col("vector").alias("rv"), F.col("norm").alias("rn"))
        scored = left.crossJoin(right).filter(F.col("left_user") != F.col("right_user")) \
            .withColumn("score", F.aggregate(F.zip_with("lv", "rv", lambda x, y: x * y), F.lit(0.0),
                                             lambda total, value: total + value) / (F.col("ln") * F.col("rn"))) \
            .withColumn("scene", F.lit(scene))
        ranked = Window.partitionBy("scene", "left_user").orderBy(F.desc("score"), F.asc("right_user"))
        results.append(scored.withColumn("rank", F.row_number().over(ranked))
                       .filter(F.col("rank") <= size).select("scene", "left_user", "right_user", "score"))
    if not results:
        return events.sparkSession.createDataFrame(
            [], "scene string, left_user string, right_user string, score double")
    result = results[0]
    for extra in results[1:]: result = result.unionByName(extra)
    return result


def item_seq_emb(events, vector_size=10, min_count=5, window_size=5, max_iter=3,
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
