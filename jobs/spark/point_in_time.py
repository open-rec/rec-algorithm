"""Distributed point-in-time feature materialization for rank training."""

from pyspark.sql import Window, functions as F


DAY_SECONDS = 86400
WINDOW_DAYS = (1, 7, 30)
EVENT_TYPES = ("click", "expose", "buy", "collect", "stay")


def _event_key(frame):
    stable = F.when(F.length(F.trim(F.col("event_id"))) > 0, F.col("event_id"))
    fallback = F.sha2(F.concat_ws("|", "user_id", "item_id", "scene", "type",
                                  F.col("time").cast("string"), "trace_id"), 256)
    return F.coalesce(stable, F.col("id") if "id" in frame.columns else fallback, fallback)


def _as_of_entities(labels, history, label_key, prefix):
    joined = labels.alias("l").join(
        history.alias("h"),
        (F.col("l.%s" % label_key) == F.col("h.id"))
        & (F.col("h._effective_time") <= F.col("l._label_time")), "inner")
    delete_order = F.when(F.upper(F.col("h._operation")) == "DELETE", 1).otherwise(0)
    window = Window.partitionBy(F.col("l._sample_id")).orderBy(
        F.desc("h._effective_time"), F.desc(delete_order), F.desc("h.dt"))
    profile_columns = [name for name in history.columns
                       if name not in {"_effective_time", "_mutation_time", "_operation", "dt"}]
    return joined.withColumn("_row", F.row_number().over(window)).filter(
        (F.col("_row") == 1) & (F.upper(F.col("h._operation")) != "DELETE")) \
        .select(F.col("l._sample_id"), *[
            F.col("h.%s" % name).alias(name) for name in profile_columns])


def _behavior(labels, history, entity_key, counterpart_key):
    keyed = history.withColumn("_event_key", _event_key(history))
    joined = labels.alias("l").join(
        keyed.alias("e"),
        (F.col("l.%s" % entity_key) == F.col("e.%s" % entity_key))
        & (F.col("e._effective_time") <= F.col("l._label_time"))
        & (F.col("e.time") < F.col("l._label_time")), "left")
    delete_order = F.when(F.upper(F.col("e._operation")) == "DELETE", 1).otherwise(0)
    latest = Window.partitionBy("l._sample_id", "e._event_key").orderBy(
        F.desc("e._effective_time"), F.desc(delete_order), F.desc("e.dt"))
    visible = joined.withColumn("_row", F.row_number().over(latest)).filter(
        (F.col("_row") == 1) & F.col("e._event_key").isNotNull()
        & (F.upper(F.col("e._operation")) != "DELETE"))
    value = F.coalesce(F.col("e.value").cast("double"), F.lit(0.0))
    expressions = [
        F.count("e._event_key").cast("double").alias("event_count"),
        F.sum(value).alias("event_value_sum"), F.avg(value).alias("event_value_mean"),
        F.countDistinct(F.floor(F.col("e.time") / DAY_SECONDS)).cast("double")
            .alias("event_active_days"),
        F.countDistinct(F.when(F.length(F.trim(F.col("e.scene"))) > 0,
                               F.col("e.scene"))).cast("double")
            .alias("event_unique_scene_count"),
        F.countDistinct("e.%s" % counterpart_key).cast("double")
            .alias("event_unique_%s_count" % counterpart_key.replace("_id", "")),
        F.min("e.time").cast("double").alias("event_first_time"),
        F.max("e.time").cast("double").alias("event_last_time"),
    ]
    for days in WINDOW_DAYS:
        expressions.append(F.sum(F.when(
            F.col("e.time") >= F.col("l._label_time") - days * DAY_SECONDS, 1).otherwise(0))
            .cast("double").alias("event_count_%dd" % days))
    for event_type in EVENT_TYPES:
        expressions.append(F.sum(F.when(F.col("e.type") == event_type, 1).otherwise(0))
                           .cast("double").alias("event_%s_count" % event_type))
    result = visible.groupBy("l._sample_id", "l._label_time").agg(*expressions)
    result = result.withColumn(
        "event_recency_seconds",
        (F.col("_label_time") - F.col("event_last_time")).cast("double"))
    denominator = F.col("event_click_count") + F.col("event_expose_count")
    return result.withColumn("event_click_rate", F.when(
        denominator > 0, F.col("event_click_count") / denominator).otherwise(0.0)) \
        .drop("_label_time")


def _enrich(labels, history, events, label_key, entity_key, counterpart_key):
    profiles = _as_of_entities(labels, history, label_key, entity_key)
    behavior = _behavior(labels, events, entity_key, counterpart_key)
    result = profiles.join(behavior, "_sample_id", "left")
    behavior_columns = [name for name in result.columns if name.startswith("event_")]
    return result.fillna(0.0, subset=behavior_columns)


def materialize_point_in_time_samples_spark(labels, event_history, user_history, item_history,
                                             target_type="item"):
    """Return labels and aligned feature rows, entirely as distributed Spark DataFrames."""
    if target_type not in ("item", "user"):
        raise ValueError("target_type must be item or user")
    fallback_identity = F.sha2(F.concat_ws(
        "|", "user_id", "item_id", "type", F.col("time"), "trace_id"), 256)
    identity = (F.coalesce(F.when(F.length(F.trim(F.col("event_id"))) > 0,
                                  F.col("event_id")), fallback_identity)
                if "event_id" in labels.columns else fallback_identity)
    prepared = labels.filter(F.col("type").isin("click", "expose")) \
        .withColumn("_sample_id", identity).withColumn("_label_time", F.col("time"))
    user_rows = _enrich(prepared, user_history, event_history,
                        "user_id", "user_id", "item_id")
    candidate_history = user_history if target_type == "user" else item_history
    candidate_entity = "user_id" if target_type == "user" else "item_id"
    candidate_rows = _enrich(prepared, candidate_history, event_history,
                             "item_id", candidate_entity,
                             "item_id" if target_type == "user" else "user_id")
    valid = user_rows.select("_sample_id").join(
        candidate_rows.select("_sample_id"), "_sample_id", "inner")
    return (prepared.join(valid, "_sample_id", "inner").drop("_label_time"),
            user_rows.join(valid, "_sample_id", "inner"),
            candidate_rows.join(valid, "_sample_id", "inner"))
