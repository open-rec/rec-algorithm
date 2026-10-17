"""Exit successfully when a model bundle is complete and matches its raw CSV inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from algorithm.feature.feature_catalog import FeatureCatalog

SCHEMA_VERSION = 1
BUILD_VERSION = 19
REQUIRED = tuple(
    f"rank/{target}/{family}/{name}"
    for target in ("item", "user")
    for family in ("lr", "fm", "lightgbm")
    for name in (f"{family}.{'txt' if family == 'lightgbm' else 'pth'}",
                 f"{family}.features.json", f"{family}.manifest.json", "user_feature.csv")
                 + (("item_feature.csv",) if target == "item" else ())
) + tuple(f"recall/item/{name}.csv" for name in (
    "item_cf_i2i", "content_i2i", "user_cf_u2i", "hot", "new", "item_seq_emb",
    "text_search",
)) + tuple(f"recall/user/{name}.csv" for name in (
    "user_cf_u2u", "content_u2u", "user_als_u2u", "user_als_emb", "text_search",
))


OPTIONAL = ("recall/item/item_title_emb.csv",)


def output_names_valid(outputs):
    return set(REQUIRED) <= set(outputs) <= set(REQUIRED + OPTIONAL)


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
        catalog = FeatureCatalog.load()
        inputs = {name: sha256(data_dir / name)
                  for name in ("user.csv", "item.csv", "event.csv")}
        return (manifest.get("schema_version") == SCHEMA_VERSION
                and manifest.get("build_version") == BUILD_VERSION
                and manifest.get("inputs") == inputs
                and manifest.get("catalog_version") == catalog.version
                and manifest.get("catalog_sha256") == catalog.sha256
                and output_names_valid(manifest.get("outputs", {}))
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
