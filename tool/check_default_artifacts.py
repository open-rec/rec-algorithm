"""Exit successfully when a model bundle is complete and matches its raw CSV inputs."""

import argparse
import hashlib
import json
from pathlib import Path

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


def valid(data_dir, model_root):
    data_dir, model_root = Path(data_dir), Path(model_root)
    path = model_root / "default.manifest.json"
    if not path.is_file() or any(not (model_root / name).is_file() for name in REQUIRED):
        return False
    try:
        manifest = json.loads(path.read_text())
        inputs = {name: sha256(data_dir / name)
                  for name in ("user.csv", "item.csv", "event.csv")}
        return (manifest.get("schema_version") == SCHEMA_VERSION
                and manifest.get("build_version") == BUILD_VERSION
                and manifest.get("inputs") == inputs
                and set(manifest.get("outputs", {})) == set(REQUIRED)
                and all(sha256(model_root / name) == digest
                        for name, digest in manifest["outputs"].items()))
    except (OSError, ValueError, KeyError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if valid(args.data, args.model_root) else 1)


if __name__ == "__main__":
    main()
