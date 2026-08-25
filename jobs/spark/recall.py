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


def content_i2i(items, cut_size=20, content_columns=("category", "tags", "title")):
    """TF-IDF cosine item similarity using category, tags and whitespace-delimited title terms."""
    available = [column for column in content_columns if column in items.columns]
    frame = items.filter(F.col("scene").isNotNull() & F.col("id").isNotNull()) \
        .dropDuplicates(["scene", "id"])
    def prefixed_tokens(column):
        values = F.split(F.lower(F.coalesce(F.col(column).cast("string"), F.lit(""))),
                         r"[,/|\s]+")
        prefix = column + ":"
        return F.transform(values, lambda token: F.concat(F.lit(prefix), token))

    token_arrays = [prefixed_tokens(column) for column in available]
    if not token_arrays:
        return items.sparkSession.createDataFrame(
            [], "scene string, left_item string, right_item string, score double")
    tokens = frame.select("scene", F.col("id").alias("item"),
                          F.array_distinct(F.flatten(F.array(*token_arrays))).alias("tokens")) \
        .select("scene", "item", F.explode("tokens").alias("token")) \
        .filter(~F.col("token").rlike(":$"))
    document_counts = tokens.select("scene", "item").distinct().groupBy("scene").count() \
        .withColumnRenamed("count", "document_count")
    frequencies = tokens.groupBy("scene", "token").count().withColumnRenamed("count", "df")
    weighted = tokens.join(document_counts, "scene").join(frequencies, ["scene", "token"]) \
        .withColumn("weight", F.log((F.col("document_count") + 1.0) /
                                    (F.col("df") + 1.0)) + 1.0)
    norms = weighted.groupBy("scene", "item").agg(
        F.sqrt(F.sum(F.col("weight") * F.col("weight"))).alias("norm"))
    left = weighted.select("scene", "token", F.col("item").alias("left_item"),
                           F.col("weight").alias("left_weight"))
    right = weighted.select("scene", "token", F.col("item").alias("right_item"),
                            F.col("weight").alias("right_weight"))
    dots = left.join(right, ["scene", "token"]).filter(F.col("left_item") != F.col("right_item")) \
        .groupBy("scene", "left_item", "right_item").agg(
            F.sum(F.col("left_weight") * F.col("right_weight")).alias("dot"))
    left_norm = norms.select("scene", F.col("item").alias("left_item"),
                             F.col("norm").alias("left_norm"))
    right_norm = norms.select("scene", F.col("item").alias("right_item"),
                              F.col("norm").alias("right_norm"))
    scored = dots.join(left_norm, ["scene", "left_item"]) \
        .join(right_norm, ["scene", "right_item"]) \
        .withColumn("score", F.col("dot") / (F.col("left_norm") * F.col("right_norm")))
    ranked = Window.partitionBy("scene", "left_item").orderBy(F.desc("score"), F.asc("right_item"))
    return scored.withColumn("rank", F.row_number().over(ranked)).filter(F.col("rank") <= cut_size) \
        .select("scene", "left_item", "right_item", "score")


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
