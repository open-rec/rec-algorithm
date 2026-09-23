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
    for suffix, seconds in (("5m", 300), ("1h", 3600), ("24h", 86400)):
        recent = F.col("e.time") >= F.col("l._label_time") - seconds
        expressions.extend((
            F.sum(F.when(recent & (F.col("e.type") == "expose"), 1).otherwise(0))
                .cast("double").alias("event_expose_count_%s" % suffix),
            F.sum(F.when(recent, value).otherwise(0.0))
                .cast("double").alias("event_value_sum_%s" % suffix),
        ))
    for event_type in EVENT_TYPES:
        expressions.append(F.sum(F.when(F.col("e.type") == event_type, 1).otherwise(0))
                           .cast("double").alias("event_%s_count" % event_type))
    for days in WINDOW_DAYS:
        in_window = F.col("e.time") >= F.col("l._label_time") - days * DAY_SECONDS
        for event_type in ("expose", "click", "collect", "buy"):
            expressions.append(F.sum(F.when(
                in_window & (F.col("e.type") == event_type), 1).otherwise(0))
                .cast("double").alias("_event_%s_count_%dd" % (event_type, days)))
    result = visible.groupBy("l._sample_id", "l._label_time").agg(*expressions)
    result = result.withColumn(
        "event_recency_seconds",
        (F.col("_label_time") - F.col("event_last_time")).cast("double"))
    denominator = F.col("event_click_count") + F.col("event_expose_count")
    result = result.withColumn("event_click_rate", F.when(
        denominator > 0, F.col("event_click_count") / denominator).otherwise(0.0)) \
        .withColumn("event_ctr", _ratio("event_click_count", "event_expose_count")) \
        .withColumn("event_collect_per_click", _ratio(
            "event_collect_count", "event_click_count")) \
        .withColumn("event_buy_per_click", _ratio("event_buy_count", "event_click_count")) \
        .withColumn("event_buy_per_collect", _ratio(
            "event_buy_count", "event_collect_count"))
    for days in WINDOW_DAYS:
        suffix = "%dd" % days
        result = result.withColumn("event_ctr_" + suffix, _ratio(
            "_event_click_count_" + suffix, "_event_expose_count_" + suffix)) \
            .withColumn("event_collect_per_click_" + suffix, _ratio(
                "_event_collect_count_" + suffix, "_event_click_count_" + suffix)) \
            .withColumn("event_buy_per_click_" + suffix, _ratio(
                "_event_buy_count_" + suffix, "_event_click_count_" + suffix)) \
            .withColumn("event_buy_per_collect_" + suffix, _ratio(
                "_event_buy_count_" + suffix, "_event_collect_count_" + suffix))
    return result.drop("_label_time", *[
        "_event_%s_count_%dd" % (event_type, days)
        for days in WINDOW_DAYS for event_type in ("expose", "click", "collect", "buy")
    ])


def _ratio(numerator, denominator):
    return F.when(F.col(denominator) > 0,
                  F.col(numerator) / F.col(denominator)).otherwise(F.lit(0.0))


def _enrich(labels, history, events, label_key, entity_key, counterpart_key):
    profiles = _as_of_entities(labels, history, label_key, entity_key)
    behavior = _behavior(labels, events, entity_key, counterpart_key)
    result = profiles.join(behavior, "_sample_id", "left")
    behavior_columns = [name for name in result.columns if name.startswith("event_")]
    return result.fillna(0.0, subset=behavior_columns)


def _commerce(labels, event_history, item_history):
    """Build user commerce features with the Item version visible at event time."""
    keyed = event_history.withColumn("_event_key", _event_key(event_history))
    joined = labels.alias("l").join(
        keyed.alias("e"),
        (F.col("l.user_id") == F.col("e.user_id"))
        & (F.col("e._effective_time") <= F.col("l._label_time"))
        & (F.col("e.time") < F.col("l._label_time")), "left")
    event_delete_order = F.when(F.upper(F.col("e._operation")) == "DELETE", 1).otherwise(0)
    latest_event = Window.partitionBy("l._sample_id", "e._event_key").orderBy(
        F.desc("e._effective_time"), F.desc(event_delete_order), F.desc("e.dt"))
    visible = joined.withColumn("_event_row", F.row_number().over(latest_event)).filter(
        (F.col("_event_row") == 1) & F.col("e._event_key").isNotNull()
        & (F.upper(F.col("e._operation")) != "DELETE"))

    contextual = visible.join(
        item_history.alias("p"),
        (F.col("e.item_id") == F.col("p.id"))
        & (F.col("p._effective_time") <= F.col("e.time")), "left")
    item_delete_order = F.when(F.upper(F.col("p._operation")) == "DELETE", 1).otherwise(0)
    latest_item = Window.partitionBy("l._sample_id", "e._event_key").orderBy(
        F.desc("p._effective_time"), F.desc(item_delete_order), F.desc("p.dt"))
    contextual = contextual.withColumn("_item_row", F.row_number().over(latest_item)).filter(
        (F.col("_item_row") == 1)
        & (F.col("p.id").isNull() | (F.upper(F.col("p._operation")) != "DELETE")))

    ext_fields = F.col("p.ext_fields") if "ext_fields" in item_history.columns \
        else F.lit(None).cast("string")
    prices = []
    for name in ("price", "unit_price", "unitPrice"):
        if name in item_history.columns:
            prices.append(F.col("p.%s" % name).cast("double"))
    prices.extend((
        F.get_json_object(ext_fields, "$.price").cast("double"),
        F.get_json_object(ext_fields, "$.unitPrice").cast("double"),
        F.get_json_object(ext_fields, "$.unit_price").cast("double"),
    ))
    event_ext = F.col("e.ext_fields") if "ext_fields" in event_history.columns \
        else F.lit(None).cast("string")
    frozen_price = F.get_json_object(
        event_ext, "$._openrecItemContext.price").cast("double")
    effective_price = F.coalesce(frozen_price, *prices)
    contextual = contextual.withColumn(
        "_item_price", F.when(effective_price >= 0, effective_price))
    expressions = [
        F.avg("_item_price").alias("event_price_mean"),
        F.stddev_pop("_item_price").alias("event_price_std"),
    ]
    for event_type in ("expose", "click", "collect", "buy"):
        selected = F.when(F.col("e.type") == event_type, F.col("_item_price"))
        expressions.extend((
            F.avg(selected).alias("event_%s_price_mean" % event_type),
            F.stddev_pop(selected).alias("event_%s_price_std" % event_type),
        ))
    for days in WINDOW_DAYS:
        recent = F.col("e.time") >= F.col("l._label_time") - days * DAY_SECONDS
        for event_type in ("click", "buy"):
            expressions.append(F.avg(F.when(
                recent & (F.col("e.type") == event_type), F.col("_item_price")))
                .alias("event_%s_price_mean_%dd" % (event_type, days)))
    result = contextual.groupBy("l._sample_id").agg(*expressions)
    result = result.withColumn("event_buy_to_click_price_ratio", _ratio(
        "event_buy_price_mean", "event_click_price_mean")) \
        .withColumn("event_recent_to_long_click_price_ratio", _ratio(
            "event_click_price_mean_1d", "event_click_price_mean_30d"))

    weights = (F.when(F.col("e.type") == "expose", 0.1)
               .when(F.col("e.type") == "click", 1.0)
               .when(F.col("e.type") == "collect", 3.0)
               .when(F.col("e.type") == "buy", 5.0)
               .when(F.col("e.type") == "stay",
                     F.log1p(F.greatest(F.coalesce(F.col("e.value").cast("double"), F.lit(0.0)),
                                        F.lit(0.0))))
               .otherwise(0.0))
    for field, output in (("category", "preferred_categories"),
                          ("subcategory", "preferred_subcategories")):
        frozen_field = F.get_json_object(
            event_ext, "$._openrecItemContext.%s" % field)
        item_field = F.col("p.%s" % field) if field in item_history.columns else F.lit(None)
        contextual_field = F.coalesce(frozen_field, item_field)
        if field not in item_history.columns and "ext_fields" not in event_history.columns:
            result = result.withColumn(output, F.lit(""))
            continue
        scored = contextual.filter(F.length(F.trim(contextual_field)) > 0) \
            .groupBy("l._sample_id", contextual_field.alias("_name")) \
            .agg(F.sum(weights).alias("_weight"))
        ranked = scored.withColumn("_rank", F.row_number().over(
            Window.partitionBy("_sample_id").orderBy(F.desc("_weight"), F.asc("_name")))) \
            .filter(F.col("_rank") <= 8)
        packed = ranked.groupBy("_sample_id").agg(F.expr(
            "concat_ws(',', transform(sort_array(collect_list(named_struct(" \
            "'rank', _rank, 'value', _name))), x -> x.value))"
        ).alias(output))
        result = result.join(packed, "_sample_id", "left")
    numeric = [name for name in result.columns
               if name.startswith("event_")]
    return result.fillna(0.0, subset=numeric).fillna(
        "", subset=["preferred_categories", "preferred_subcategories"])


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
    user_rows = user_rows.join(
        _commerce(prepared, event_history, item_history), "_sample_id", "left")
    candidate_history = user_history if target_type == "user" else item_history
    candidate_entity = "user_id" if target_type == "user" else "item_id"
    candidate_rows = _enrich(prepared, candidate_history, event_history,
                             "item_id", candidate_entity,
                             "item_id" if target_type == "user" else "user_id")
    if target_type == "item":
        label_times = prepared.select("_sample_id", "_label_time")
        # coalesce handles null values, but Spark still resolves every column
        # reference. Legacy/minimal item histories may omit pub_time entirely.
        published_time = (F.col("pub_time") if "pub_time" in candidate_rows.columns
                          else F.lit(None).cast("long"))
        candidate_rows = candidate_rows.join(label_times, "_sample_id", "left") \
            .withColumn(
                "content_age_hours",
                F.greatest(
                    F.lit(0.0),
                    (
                        F.col("_label_time")
                        - F.coalesce(published_time, F.col("_label_time"))
                    )
                    / F.lit(3600.0),
                ),
            ).drop("_label_time")
    valid = user_rows.select("_sample_id").join(
        candidate_rows.select("_sample_id"), "_sample_id", "inner")
    return (prepared.join(valid, "_sample_id", "inner").drop("_label_time"),
            user_rows.join(valid, "_sample_id", "inner"),
            candidate_rows.join(valid, "_sample_id", "inner"))
