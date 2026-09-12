"""Train and evaluate a versioned rank artifact from cumulative Hive entity data."""

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import urllib.request

from pyspark.sql import SparkSession, Window, functions as F

from jobs.spark.io import read_events, read_items, read_users


def parser():
    result = argparse.ArgumentParser(description="OpenRec rank model training job")
    result.add_argument("--date", required=True)
    result.add_argument("--revision", default="r001")
    result.add_argument("--scene", default="scene_0")
    result.add_argument("--event-path", default="hdfs://namenode:8020/openrec/hive/event")
    result.add_argument("--item-path", default="hdfs://namenode:8020/openrec/hive/item")
    result.add_argument("--user-path", default="hdfs://namenode:8020/openrec/hive/user")
    result.add_argument("--artifact-root", default="/models/releases")
    result.add_argument("--epochs", type=int, default=5)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--validation-ratio", type=float, default=.2)
    result.add_argument("--min-auc", type=float, default=0.0)
    result.add_argument("--model-type", choices=("lr", "fm"), default="lr")
    result.add_argument("--target-type", choices=("item", "user"), default="item")
    result.add_argument("--factor-dim", type=int, default=8)
    result.add_argument("--max-events", type=int, default=200000)
    result.add_argument("--user-label-window-days", type=int, default=7)
    return result


def _validate(args):
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
        raise ValueError("date must use YYYY-MM-DD")
    if not re.match(r"^r\d{3,}$", args.revision):
        raise ValueError("revision must look like r001")
    if not re.match(r"^[A-Za-z0-9_-]+$", args.scene):
        raise ValueError("scene contains unsupported characters")
    if not 0 <= args.min_auc <= 1 or not 0 < args.validation_ratio < 1:
        raise ValueError("invalid evaluation threshold or validation ratio")
    if not 1 <= args.factor_dim <= 256:
        raise ValueError("factor_dim must be between 1 and 256")
    if not 1 <= args.user_label_window_days <= 30:
        raise ValueError("user_label_window_days must be between 1 and 30")


def _freeze_feature_history(all_events, labelled_events):
    """Return a strictly-prior feature population and its auditable cutoff."""
    cutoff = labelled_events.agg(F.min("time").alias("cutoff")).first()["cutoff"]
    if cutoff is None:
        raise ValueError("rank training data has no labelled events")
    return all_events.filter(F.col("time") < F.lit(cutoff)), cutoff


def _user_pair_events(events, users, max_events):
    """Build balanced point-in-time U2U labels from active users in one label window."""
    user_ids = users.select(F.col("id").cast("string").alias("id")).dropDuplicates()
    positive = events.filter(F.col("type").isin("click", "buy", "collect")) \
        .filter(F.col("user_id").isNotNull() & F.col("item_id").isNotNull()) \
        .select("scene", "user_id", "item_id", "time") \
        .join(user_ids, F.col("user_id") == F.col("id"), "left_semi") \
        .dropDuplicates()
    pairs = positive.alias("l").join(
        positive.alias("r"),
        (F.col("l.scene") == F.col("r.scene"))
        & (F.col("l.item_id") == F.col("r.item_id"))
        & (F.col("l.user_id") != F.col("r.user_id"))) \
        .select(F.col("l.user_id").alias("user_id"),
                F.col("r.user_id").alias("item_id"),
                F.greatest(F.col("l.time"), F.col("r.time")).alias("time")) \
        .groupBy("user_id", "item_id").agg(F.min("time").alias("time"))
    active_ids = positive.select(F.col("user_id").alias("id")).dropDuplicates()
    source_stats = pairs.groupBy("user_id").agg(
        F.count("*").alias("positive_count"), F.max("time").alias("label_time"))
    negative_rank = Window.partitionBy("user_id").orderBy(F.xxhash64("user_id", "item_id"))
    negatives = active_ids.alias("l").crossJoin(active_ids.alias("r")) \
        .filter(F.col("l.id") != F.col("r.id")) \
        .select(F.col("l.id").alias("user_id"), F.col("r.id").alias("item_id")) \
        .join(pairs.select("user_id", "item_id"), ["user_id", "item_id"], "left_anti") \
        .join(source_stats, "user_id") \
        .withColumn("rank", F.row_number().over(negative_rank)) \
        .filter(F.col("rank") <= F.col("positive_count")) \
        .select("user_id", "item_id", F.col("label_time").alias("time"))
    per_class_limit = max(1, max_events // 2)
    positives = pairs.orderBy(F.xxhash64("user_id", "item_id")).limit(per_class_limit) \
        .withColumn("type", F.lit("click"))
    negatives = negatives.orderBy(F.xxhash64("user_id", "item_id")).limit(per_class_limit) \
        .withColumn("type", F.lit("expose"))
    return positives.unionByName(negatives) \
        .withColumn("trace_id", F.concat_ws("-", F.lit("u2u"), "user_id", "item_id")) \
        .orderBy("time", "user_id", "item_id", "type")


def run(args, spark=None):
    _validate(args)
    spark = spark or SparkSession.builder.appName("openrec-rank-train").enableHiveSupport().getOrCreate()
    all_events = read_events(spark, date=args.date, cumulative=True, path=args.event_path)
    business_day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    label_days = args.user_label_window_days if args.target_type == "user" else 1
    label_from = int((business_day - timedelta(days=label_days - 1)).timestamp())
    label_until = int((business_day + timedelta(days=1)).timestamp())
    label_window = all_events.filter(
        (F.col("scene") == args.scene) & (F.col("time") >= label_from)
        & (F.col("time") < label_until))
    events = label_window.filter(F.col("type").isin("click", "expose")) \
        .orderBy(F.desc("time")).limit(args.max_events)

    # Freeze every feature source at one auditable instant strictly before the first label. This is
    # conservative (all samples share a snapshot) but prevents profile mutations later in the day
    # from leaking into earlier labels. Per-sample temporal joins can build on this contract later.
    provisional_cutoff = events.agg(F.min("time").alias("cutoff")).first()["cutoff"]
    if provisional_cutoff is None:
        raise ValueError("rank training data has no labelled events")
    items = read_items(spark, date=args.date, cumulative=True, path=args.item_path,
                       as_of_time=provisional_cutoff) \
        .filter(F.col("scene") == args.scene)
    users = read_users(spark, date=args.date, cumulative=True, path=args.user_path,
                       as_of_time=provisional_cutoff)
    active_events = events.join(items.select(F.col("id").alias("active_item")),
                                events.item_id == F.col("active_item"), "left_semi")
    if args.target_type == "user":
        active_events = _user_pair_events(label_window, users, args.max_events)
    feature_events, feature_cutoff_time = _freeze_feature_history(all_events, active_events)
    if feature_cutoff_time != provisional_cutoff:
        items = read_items(spark, date=args.date, cumulative=True, path=args.item_path,
                           as_of_time=feature_cutoff_time).filter(F.col("scene") == args.scene)
        users = read_users(spark, date=args.date, cumulative=True, path=args.user_path,
                           as_of_time=feature_cutoff_time)
    event_frame, feature_event_frame, item_frame, user_frame = (
        active_events.toPandas(), feature_events.toPandas(), items.toPandas(), users.toPandas())
    if event_frame.empty or user_frame.empty or (args.target_type == "item" and item_frame.empty):
        raise ValueError("rank training data is empty after active entity filtering")
    version = "%s-%s" % (args.date.replace("-", ""), args.revision)
    dataset_dir = Path(args.artifact_root).parent / "training" / args.target_type / args.scene / version
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    try:
        event_frame.to_json(dataset_dir / "events.jsonl", orient="records", lines=True)
        feature_event_frame.to_json(
            dataset_dir / "feature_events.jsonl", orient="records", lines=True)
        item_frame.to_json(dataset_dir / "items.jsonl", orient="records", lines=True)
        user_frame.to_json(dataset_dir / "users.jsonl", orient="records", lines=True)
        payload = json.dumps({"scene": args.scene, "version": version,
                              "business_date": args.date, "revision": args.revision,
                              "dataset_dir": str(dataset_dir), "epochs": args.epochs,
                              "batch_size": args.batch_size,
                              "validation_ratio": args.validation_ratio,
                              "min_auc": args.min_auc, "model_type": args.model_type,
                              "target_type": args.target_type,
                              "factor_dim": args.factor_dim,
                              "feature_cutoff_time": int(feature_cutoff_time)}).encode()
        request = urllib.request.Request(os.environ.get(
            "RANK_ENGINE_URL", "http://rank-engine:8123") + "/model/train", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.loads(response.read())
        if result.get("status") != "success":
            raise ValueError("rank-engine training failed: %s" % result.get("message"))
        manifest = result["data"]
        print("OPENREC_MODEL_MANIFEST=" + json.dumps(manifest, sort_keys=True))
        return manifest
    except Exception:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    run(parser().parse_args())
