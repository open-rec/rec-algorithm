"""Build or reuse the complete deployable ``model`` bundle from raw OpenRec CSVs."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.fm import FMRecModel
from algorithm.rank.lr import LRRecModel
from tool.gen_recall_data import generate as generate_recall


SCHEMA_VERSION = 1
BUILD_VERSION = 2
REQUIRED = (
    "feature/default/user_feature.csv", "feature/default/item_feature.csv",
    "feature/default/lr.features.json", "feature/default/fm.features.json",
    "rank/default/lr.pth", "rank/default/fm.pth",
    "rank/default/lr.manifest.json", "rank/default/fm.manifest.json",
    "recall/i2i.csv", "recall/hot.csv", "recall/new.csv", "recall/embedding.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(data_dir):
    return {name: sha256(Path(data_dir) / name) for name in ("user.csv", "item.csv", "event.csv")}


def reusable(model_root, inputs):
    root = Path(model_root)
    manifest_path = root / "default.manifest.json"
    if not manifest_path.is_file() or any(not (root / name).is_file() for name in REQUIRED):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        return (manifest.get("schema_version") == SCHEMA_VERSION
                and manifest.get("build_version") == BUILD_VERSION
                and manifest.get("inputs") == inputs
                and all(sha256(root / name) == digest
                        for name, digest in manifest.get("outputs", {}).items()))
    except (OSError, ValueError):
        return False


def _train(model_class, model_type, users, items, feature_events, labels, cutoff, stage,
           epochs, factor_dim, min_auc):
    model_file = stage / "rank/default" / (model_type + ".pth")
    feature_file = stage / "feature/default" / (model_type + ".features.json")
    kwargs = {"factor_dim": factor_dim} if model_type == "fm" else {}
    model = model_class(UserFeature(users, feature_events, cutoff),
                        ItemFeature(items, feature_events, cutoff), labels,
                        scene="default", model_file=model_file, feature_file=feature_file, **kwargs)
    if not len(model.dataset) or model.dataset.positive_rate in (0.0, 1.0):
        raise ValueError("%s training requires non-empty click and unclicked-expose labels" % model_type)
    model.train(epoch_num=epochs, batch_size=256, val_ratio=.2)
    _, validation = model._split(val_ratio=.2, seed=42)
    auc = model.evaluate(validation, batch_size=256)
    if auc is None:
        raise ValueError("%s validation AUC is undefined" % model_type)
    if auc < min_auc:
        raise ValueError("%s validation AUC %.6f is below %.6f" % (model_type, auc, min_auc))
    model.save()
    feature_space = model.dataset.feature_space
    manifest = {
        "version": "default", "scene": "default", "model_type": model_type,
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "evaluated",
        "feature_cutoff_time": cutoff, "model": model_file.name, "feature": feature_file.name,
        "feature_set": feature_space.feature_set, "catalog_version": feature_space.catalog_version,
        "feature_sha256": sha256(feature_file), "input_dim": model.model.dim,
        "metrics": {"auc": auc, "positive_rate": model.dataset.positive_rate,
                    "samples": len(model.dataset), "feature_dim": model.model.dim,
                    **({"factor_dim": model.model.factor_dim} if model_type == "fm" else {})},
        "gate": {"min_auc": min_auc, "passed": True},
    }
    (stage / "rank/default" / (model_type + ".manifest.json")).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build(data_dir, model_root, epochs=8, factor_dim=8, min_auc=.70, force=False):
    data_dir, model_root = Path(data_dir).resolve(), Path(model_root).resolve()
    inputs = input_hashes(data_dir)
    if not force and reusable(model_root, inputs):
        return {"status": "reused", "model_root": str(model_root), "inputs": inputs}
    model_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".default-build-", dir=str(model_root)))
    try:
        for relative in ("feature/default", "rank/default", "recall"):
            (stage / relative).mkdir(parents=True)
        users = pd.read_csv(data_dir / "user.csv")
        items = pd.read_csv(data_dir / "item.csv")
        events = pd.read_csv(data_dir / "event.csv")
        times = pd.to_numeric(events["time"], errors="coerce").dropna()
        if times.empty:
            raise ValueError("event.csv has no valid timestamps")
        cutoff = int(times.quantile(.8))
        feature_events = events[pd.to_numeric(events["time"], errors="coerce") < cutoff]
        labels = events[pd.to_numeric(events["time"], errors="coerce") >= cutoff]

        user_snapshot = UserFeature(users, feature_events, cutoff).users.copy()
        item_snapshot = ItemFeature(items, feature_events, cutoff).items.copy()
        user_snapshot.insert(1, "as_of_time", cutoff)
        item_snapshot.insert(1, "as_of_time", cutoff)
        user_snapshot.to_csv(stage / "feature/default/user_feature.csv", index=False)
        item_snapshot.to_csv(stage / "feature/default/item_feature.csv", index=False)
        _train(LRRecModel, "lr", users, items, feature_events, labels, cutoff, stage,
               epochs, factor_dim, min_auc)
        _train(FMRecModel, "fm", users, items, feature_events, labels, cutoff, stage,
               epochs, factor_dim, min_auc)
        generate_recall(data_dir / "item.csv", data_dir / "event.csv", stage / "recall")

        outputs = {name: sha256(stage / name) for name in REQUIRED}
        manifest = {"schema_version": SCHEMA_VERSION, "build_version": BUILD_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "source": "rec-algorithm/tool/build_default_artifacts.py",
                    "feature_cutoff_time": cutoff, "inputs": inputs, "outputs": outputs}
        for name in REQUIRED:
            target = model_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
        (model_root / "default.manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"status": "built", "model_root": str(model_root), "inputs": inputs,
                "feature_cutoff_time": cutoff}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--factor-dim", type=int, default=8)
    parser.add_argument("--min-auc", type=float, default=.70)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.data, args.model_root, args.epochs, args.factor_dim,
                           args.min_auc, args.force),
                     sort_keys=True))


if __name__ == "__main__":
    main()
