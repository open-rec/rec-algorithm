"""Train and evaluate a versioned rank artifact from cumulative Hive entity data."""

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import urllib.request

from pyspark.sql import SparkSession, functions as F

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
    result.add_argument("--factor-dim", type=int, default=8)
    result.add_argument("--max-events", type=int, default=200000)
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


def _freeze_feature_history(all_events, labelled_events):
    """Return a strictly-prior feature population and its auditable cutoff."""
    cutoff = labelled_events.agg(F.min("time").alias("cutoff")).first()["cutoff"]
    if cutoff is None:
        raise ValueError("rank training data has no labelled events")
    return all_events.filter(F.col("time") < F.lit(cutoff)), cutoff


def run(args, spark=None):
    _validate(args)
    spark = spark or SparkSession.builder.appName("openrec-rank-train").enableHiveSupport().getOrCreate()
    all_events = read_events(spark, date=args.date, cumulative=True, path=args.event_path)
    business_day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    label_from = int(business_day.timestamp())
    label_until = int((business_day + timedelta(days=1)).timestamp())
    events = all_events \
        .filter((F.col("scene") == args.scene) & F.col("type").isin("click", "expose")
                & (F.col("time") >= label_from) & (F.col("time") < label_until)) \
        .orderBy(F.desc("time")).limit(args.max_events)

    # One frozen, strictly-prior snapshot is shared by all samples in this training run. This is a
    # deliberately conservative point-in-time contract: neither a label event itself nor any later
    # train/validation event can enter its behavioural features.
    feature_events, feature_cutoff_time = _freeze_feature_history(all_events, events)
    # Entity snapshots are the latest active state within the requested business date. Behavioural
    # aggregates alone use the strictly-prior cutoff; filtering profiles by the first label time
    # would incorrectly remove entities inserted earlier on the same day but processed milliseconds
    # after a client-generated event timestamp.
    items = read_items(spark, date=args.date, cumulative=True, path=args.item_path) \
        .filter(F.col("scene") == args.scene)
    users = read_users(spark, date=args.date, cumulative=True, path=args.user_path)
    active_events = events.join(items.select(F.col("id").alias("active_item")),
                                events.item_id == F.col("active_item"), "left_semi")
    event_frame, feature_event_frame, item_frame, user_frame = (
        active_events.toPandas(), feature_events.toPandas(), items.toPandas(), users.toPandas())
    if event_frame.empty or item_frame.empty or user_frame.empty:
        raise ValueError("rank training data is empty after active entity filtering")
    version = "%s-%s" % (args.date.replace("-", ""), args.revision)
    dataset_dir = Path(args.artifact_root).parent / "training" / args.scene / version
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
