"""Distributed content recall formulas."""

from pyspark.sql import Window, functions as F


def content_u2u(users, size=100, content_columns=("gender", "country", "city", "tags")):
    """TF-IDF cosine similarity over user profile content."""
    available = [column for column in content_columns if column in users.columns]
    scene = F.col("scene") if "scene" in users.columns else F.lit("default")
    frame = users.filter(F.col("id").isNotNull()).withColumn("scene", scene) \
        .dropDuplicates(["scene", "id"])
    arrays = [F.transform(F.split(F.lower(F.coalesce(F.col(column).cast("string"), F.lit(""))),
                                  r"[,/|\s]+"),
                          lambda token: F.concat(F.lit(column + ":"), token))
              for column in available]
    if not arrays:
        return users.sparkSession.createDataFrame(
            [], "scene string, left_user string, right_user string, score double")
    tokens = frame.select("scene", F.col("id").alias("user"),
                          F.array_distinct(F.flatten(F.array(*arrays))).alias("tokens")) \
        .select("scene", "user", F.explode("tokens").alias("token")).filter(~F.col("token").rlike(":$"))
    docs = tokens.select("scene", "user").distinct().groupBy("scene").count() \
        .withColumnRenamed("count", "document_count")
    frequencies = tokens.groupBy("scene", "token").count().withColumnRenamed("count", "df")
    weighted = tokens.join(docs, "scene").join(frequencies, ["scene", "token"]) \
        .withColumn("weight", F.log((F.col("document_count") + 1.0) / (F.col("df") + 1.0)) + 1.0)
    norms = weighted.groupBy("scene", "user").agg(
        F.sqrt(F.sum(F.col("weight") * F.col("weight"))).alias("norm"))
    left = weighted.select("scene", "token", F.col("user").alias("left_user"),
                           F.col("weight").alias("left_weight"))
    right = weighted.select("scene", "token", F.col("user").alias("right_user"),
                            F.col("weight").alias("right_weight"))
    dots = left.join(right, ["scene", "token"]).filter(F.col("left_user") != F.col("right_user")) \
        .groupBy("scene", "left_user", "right_user").agg(
            F.sum(F.col("left_weight") * F.col("right_weight")).alias("dot"))
    left_norm = norms.select("scene", F.col("user").alias("left_user"), F.col("norm").alias("ln"))
    right_norm = norms.select("scene", F.col("user").alias("right_user"), F.col("norm").alias("rn"))
    scored = dots.join(left_norm, ["scene", "left_user"]).join(right_norm, ["scene", "right_user"]) \
        .withColumn("score", F.col("dot") / (F.col("ln") * F.col("rn")))
    ranked = Window.partitionBy("scene", "left_user").orderBy(F.desc("score"), F.asc("right_user"))
    return scored.withColumn("rank", F.row_number().over(ranked)).filter(F.col("rank") <= size) \
        .select("scene", "left_user", "right_user", "score")


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


def text_search_documents(items, text_columns=("title", "category", "tags"), entity_field="item"):
    """Build one generic full-text search document per scene-scoped item or user."""
    available = [column for column in text_columns if column in items.columns]
    frame = items.filter(F.col("scene").isNotNull() & F.col("id").isNotNull()) \
        .dropDuplicates(["scene", "id"])
    if not available:
        return frame.select("scene", F.col("id").alias(entity_field)) \
            .withColumn("text", F.lit(""))
    values = [F.coalesce(F.col(column).cast("string"), F.lit("")) for column in available]
    return frame.select("scene", F.col("id").alias(entity_field),
                        F.concat_ws(" ", *values).alias("text"))
