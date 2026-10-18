"""Build the rank Spark submission without importing PySpark."""

import os
import re
import json

from jobs.spark.common.submission import spark_submit


def rank_command(payload):
    from algorithm.feature.feature_catalog import select_training_features

    business_date = payload.get("date")
    revision = payload.get("revision", "r001")
    scene = payload.get("scene", "scene_0")
    model_type = payload.get("model_type", "lr")
    target_type = payload.get("target_type", "item")
    factor_dim = int(payload.get("factor_dim", 8))
    user_label_window_days = int(payload.get("user_label_window_days", 7))
    max_events = int(payload.get("max_events", 200000))
    max_history_rows = int(payload.get("max_history_rows", 5000000))
    max_materialization_seconds = int(
        payload.get("max_materialization_seconds", 1800)
    )
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", business_date or ""):
        raise ValueError("date must use YYYY-MM-DD")
    if not re.match(r"^r\d{3,}$", revision):
        raise ValueError("revision must look like r001")
    if not re.match(r"^[A-Za-z0-9_-]+$", scene):
        raise ValueError("invalid scene")
    if model_type not in ("lr", "fm", "lightgbm"):
        raise ValueError("model_type must be lr, fm or lightgbm")
    if target_type not in ("item", "user"):
        raise ValueError("target_type must be item or user")
    if not 1 <= factor_dim <= 256:
        raise ValueError("factor_dim must be between 1 and 256")
    if not 1 <= user_label_window_days <= 30:
        raise ValueError("user_label_window_days must be between 1 and 30")
    if (
        max_events < 1
        or max_history_rows < max_events
        or max_materialization_seconds < 1
    ):
        raise ValueError("invalid PIT materialization resource gate")
    selection = select_training_features(
        model_type, target_type, payload.get("feature_selection")
    )
    return spark_submit("rank", "ranking/job.py") + [
        "--date",
        business_date,
        "--revision",
        revision,
        "--scene",
        scene,
        "--event-path",
        os.environ.get(
            "OPENREC_EVENT_PATH", "hdfs://namenode:8020/openrec/hive/event"
        ),
        "--item-path",
        os.environ.get(
            "OPENREC_ITEM_PATH", "hdfs://namenode:8020/openrec/hive/item"
        ),
        "--user-path",
        os.environ.get(
            "OPENREC_USER_PATH", "hdfs://namenode:8020/openrec/hive/user"
        ),
        "--artifact-root",
        os.environ.get("MODEL_ARTIFACT_ROOT", "/models/releases"),
        "--epochs",
        str(payload.get("epochs", 5)),
        "--min-auc",
        str(payload.get("min_auc", 0.0)),
        "--model-type",
        model_type,
        "--target-type",
        target_type,
        "--factor-dim",
        str(factor_dim),
        "--feature-selection",
        json.dumps(selection),
        "--batch-size",
        str(payload.get("batch_size", 256)),
        "--validation-ratio",
        str(payload.get("validation_ratio", 0.2)),
        "--user-label-window-days",
        str(user_label_window_days),
        "--max-events",
        str(max_events),
        "--max-history-rows",
        str(max_history_rows),
        "--max-materialization-seconds",
        str(max_materialization_seconds),
    ]
