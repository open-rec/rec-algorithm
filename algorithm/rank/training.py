"""Offline rank training and immutable artifact creation.

Executed by the rec-algorithm Spark job, never by the inference service.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from pydantic import BaseModel, Field

from algorithm.feature.feature_catalog import select_features
from algorithm.feature.feature_space import FeatureSpace
from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.point_in_time import materialize_point_in_time_samples
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.fm import FMRecModel
from algorithm.rank.lr import LRRecModel


class TrainingRequest(BaseModel):
    model_config = {"extra": "forbid"}
    feature_selection: Optional[dict[str, List[str]]] = None
    scene: str = Field(pattern="^[A-Za-z0-9_-]+$")
    version: str = Field(pattern="^[A-Za-z0-9_-]+$")
    business_date: str
    revision: str
    dataset_dir: str
    epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=256, ge=1)
    validation_ratio: float = Field(default=0.2, gt=0, lt=1)
    min_auc: float = Field(default=0.0, ge=0, le=1)
    model_type: str = Field(default="lr", pattern="^(lr|fm)$")
    factor_dim: int = Field(default=8, ge=1, le=256)
    label_observation_cutoff: int = Field(ge=0)
    input_label_count: int = Field(ge=0)
    constructed_label_count: Optional[int] = Field(default=None, ge=0)
    materialized_label_count: Optional[int] = Field(default=None, ge=0)
    history_row_count: Optional[int] = Field(default=None, ge=0)
    materialization_seconds: Optional[float] = Field(default=None, ge=0)
    feature_cutoff_time: int = Field(ge=0)
    feature_until_time: Optional[int] = Field(default=None, ge=0)
    target_type: str = Field(default="item", pattern="^(item|user)$")


def _align_materialized_rows(events, sample_users, sample_items):
    """Align Spark materialized rows by immutable sample identity."""
    frames = (
        ("events", events),
        ("sample_users", sample_users),
        ("sample_items", sample_items),
    )
    for name, frame in frames:
        if "_sample_id" not in frame.columns:
            raise ValueError("%s is missing _sample_id" % name)
        if (
            frame["_sample_id"].isna().any()
            or frame["_sample_id"].duplicated().any()
        ):
            raise ValueError("%s contains null or duplicate _sample_id" % name)
    expected = set(events["_sample_id"])
    if (
        set(sample_users["_sample_id"]) != expected
        or set(sample_items["_sample_id"]) != expected
    ):
        raise ValueError(
            "materialized feature rows do not match event sample identities"
        )
    order = events["_sample_id"].tolist()
    return (
        sample_users.set_index("_sample_id").loc[order].reset_index(),
        sample_items.set_index("_sample_id").loc[order].reset_index(),
    )


def _latest_feature_rows(events, rows):
    """Select latest per-entity PIT rows regardless of part-file order."""
    if "_sample_id" in events.columns and "_sample_id" in rows.columns:
        timeline = events[["_sample_id", "time"]].rename(
            columns={"time": "_label_time"}
        )
        ordered = rows.merge(timeline, on="_sample_id", validate="one_to_one")
    else:
        if len(events) != len(rows):
            raise ValueError("feature rows are not aligned with label events")
        ordered = rows.copy()
        ordered["_label_time"] = events["time"].tolist()
    ordered["_sample_order"] = range(len(ordered))
    ordered = ordered.sort_values(
        ["_label_time", "_sample_order"], kind="mergesort"
    )
    return (
        ordered.drop_duplicates("id", keep="last")
        .drop(
            columns=["_sample_id", "_label_time", "_sample_order"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )


def train_release(info: TrainingRequest, artifact_root):
    """Train on CPU and atomically retain an evaluated, immutable release."""
    threads = int(os.environ.get("RANK_TRAINING_THREADS", "2"))
    if not 1 <= threads <= 256:
        raise ValueError("RANK_TRAINING_THREADS must be between 1 and 256")
    torch.set_num_threads(threads)
    selection = select_features(
        info.model_type, info.target_type, info.feature_selection
    )
    dataset = Path(info.dataset_dir).resolve()
    artifact_root = Path(artifact_root).resolve()
    training_root = artifact_root.parent / "training"
    if training_root not in dataset.parents or not re.match(
        r"^[A-Za-z0-9_-]+$", info.scene
    ):
        raise ValueError("dataset must be inside the training directory")
    target = artifact_root / info.target_type / info.scene / info.version
    if target.exists():
        raise ValueError("model version already exists")
    scene_root = artifact_root / info.target_type / info.scene
    scene_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".%s-" % info.version, dir=str(scene_root))
    )
    try:

        def read_records(path):
            files = (
                sorted(path.glob("part-*.json")) if path.is_dir() else [path]
            )
            frames = [
                pd.read_json(value, lines=True)
                for value in files
                if value.stat().st_size
            ]
            return (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame()
            )

        events = read_records(dataset / "events.jsonl")
        model_type = info.model_type.strip().lower()
        model_filename = "%s.pth" % model_type
        feature_filename = "%s.features.json" % model_type
        model_class = {"lr": LRRecModel, "fm": FMRecModel}[model_type]
        model_kwargs = (
            {"factor_dim": info.factor_dim} if model_type == "fm" else {}
        )
        if (dataset / "sample_users.jsonl").exists():
            sample_users = read_records(dataset / "sample_users.jsonl")
            sample_items = read_records(dataset / "sample_items.jsonl")
            sample_users, sample_items = _align_materialized_rows(
                events, sample_users, sample_items
            )
        else:
            feature_events = read_records(dataset / "feature_events.jsonl")
            items = read_records(dataset / "items.jsonl")
            users = read_records(dataset / "users.jsonl")
            events, sample_users, sample_items = (
                materialize_point_in_time_samples(
                    events, feature_events, users, items, info.target_type
                )
            )
        if events.empty:
            raise ValueError(
                "rank training has no entities active at their label times"
            )
        space = FeatureSpace.for_model(
            info.model_type, info.target_type, selection
        )
        for frame, columns in (
            (sample_users, space.user_columns),
            (sample_items, space.item_columns),
        ):
            for column in columns:
                if (
                    column.name not in frame
                    or not frame[column.name].notna().any()
                ):
                    raise ValueError(
                        "offline feature has no materialized values: %s"
                        % column.feature_id
                    )
        latest_users = _latest_feature_rows(events, sample_users)
        latest_items = _latest_feature_rows(events, sample_items)
        user_features = UserFeature(latest_users)
        item_features = (
            ItemFeature(latest_items)
            if info.target_type == "item"
            else UserFeature(latest_items)
        )
        rank_model = model_class(
            user_features,
            item_features,
            events,
            feature_space=space,
            scene=info.scene,
            model_file=staging / model_filename,
            feature_file=staging / feature_filename,
            target_type=info.target_type,
            sample_users=sample_users,
            sample_items=sample_items,
            validation_ratio=info.validation_ratio,
            **model_kwargs,
        )
        if not len(rank_model.dataset):
            raise ValueError(
                "rank training produced no labelled samples "
                "after entity filtering"
            )
        if rank_model.dataset.positive_rate in (0.0, 1.0):
            raise ValueError(
                "rank training requires both click and expose labels"
            )
        training, validation = rank_model._split(
            val_ratio=info.validation_ratio, seed=42
        )
        rank_model.train(
            epoch_num=info.epochs,
            batch_size=info.batch_size,
            val_ratio=info.validation_ratio,
        )
        auc = rank_model.evaluate(validation, batch_size=info.batch_size)
        if auc is None:
            raise ValueError("AUC is undefined for validation data")
        if auc is not None and auc < info.min_auc:
            raise ValueError("AUC %.6f is below %.6f" % (auc, info.min_auc))
        rank_model.save()
        # Keep the unencoded entity snapshots next to the checkpoint. They are
        # the portable
        # bootstrap representation for Redis; *.features.json remains the
        # model-specific encoding
        # contract and must not be confused with actual entity feature values.
        candidate_features = (
            item_features.items
            if info.target_type == "item"
            else item_features.users
        )
        for frame, filename in (
            (user_features.users, "user_feature.csv"),
            (candidate_features, "item_feature.csv"),
        ):
            exported = frame.copy()
            exported.insert(
                1,
                "as_of_time",
                info.feature_until_time or info.feature_cutoff_time,
            )
            exported.to_csv(staging / filename, index=False)
        feature_bytes = (staging / feature_filename).read_bytes()
        feature_sha256 = hashlib.sha256(feature_bytes).hexdigest()
        feature_space = rank_model.dataset.feature_space
        manifest = {
            "model_sha256": hashlib.sha256(
                (staging / model_filename).read_bytes()
            ).hexdigest(),
            "feature_selection": feature_space.selection,
            "feature_definitions": feature_space.feature_definitions,
            "training_config": {
                "epochs": info.epochs,
                "batch_size": info.batch_size,
                "validation_ratio": info.validation_ratio,
                "min_auc": info.min_auc,
                "model_type": info.model_type,
                "factor_dim": info.factor_dim,
                "scene": info.scene,
                "target_type": info.target_type,
            },
            "version": info.version,
            "scene": info.scene,
            "model_type": model_type,
            "target_type": info.target_type,
            "business_date": info.business_date,
            "revision": info.revision,
            "label_observation_cutoff": info.label_observation_cutoff,
            "feature_cutoff_time": info.feature_cutoff_time,
            "feature_until_time": info.feature_until_time
            or info.feature_cutoff_time,
            "feature_join": "per_sample_point_in_time",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "evaluated",
            "model": model_filename,
            "feature": feature_filename,
            "user_feature_snapshot": "user_feature.csv",
            "item_feature_snapshot": "item_feature.csv",
            "feature_set": feature_space.feature_set,
            "catalog_version": feature_space.catalog_version,
            "catalog_sha256": feature_space.catalog_sha256,
            "feature_sha256": feature_sha256,
            "input_dim": rank_model.model.dim,
            "metrics": {
                "auc": auc,
                "positive_rate": rank_model.dataset.positive_rate,
                "samples": len(rank_model.dataset),
                "input_labels": info.input_label_count,
                "constructed_labels": info.constructed_label_count,
                "spark_materialized_labels": info.materialized_label_count,
                "dropped_labels": (
                    (
                        info.constructed_label_count
                        if info.constructed_label_count is not None
                        else info.input_label_count
                    )
                    - len(rank_model.dataset)
                ),
                "history_rows": info.history_row_count,
                "materialization_seconds": info.materialization_seconds,
                "training_samples": len(training),
                "validation_samples": len(validation),
                "label_time_min": int(events["time"].min()),
                "label_time_max": int(events["time"].max()),
                "feature_dim": rank_model.model.dim,
                **(
                    {"factor_dim": rank_model.model.factor_dim}
                    if model_type == "fm"
                    else {}
                ),
            },
            "gate": {"min_auc": info.min_auc, "passed": True},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True)
        )
        os.replace(staging, target)
        shutil.rmtree(dataset, ignore_errors=True)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    info = TrainingRequest.model_validate_json(Path(args.request).read_text())
    train_release(info, args.artifact_root)


if __name__ == "__main__":
    main()
