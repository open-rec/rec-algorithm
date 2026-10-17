"""Migrate a verified v11–v16 bootstrap bundles to v17 without retraining rank weights."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from tool.check_default_artifacts import BUILD_VERSION, REQUIRED, sha256, valid
from tool.gen_recall_data import generate
from tool.gen_text_embedding_data import generate as generate_text_embeddings


def migrate(data_dir, model_root):
    data_dir, root = Path(data_dir), Path(model_root)
    if valid(data_dir, root):
        return
    manifest = json.loads((root / "default.manifest.json").read_text())
    if manifest["build_version"] not in (11, 12, 13, 14, 15, 16):
        raise ValueError("migration requires a v11–v16 bundle; otherwise use build_default_artifacts")
    for name, digest in manifest["inputs"].items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f"input checksum mismatch: {name}")
    for name, digest in manifest["outputs"].items():
        # This obsolete model-specific sidecar may already have been removed.
        if name == "recall/item/text_embedding.manifest.json" and not (root / name).exists():
            continue
        if sha256(root / name) != digest:
            raise ValueError(f"output checksum mismatch: {name}")
    with tempfile.TemporaryDirectory(prefix=".layout-migration-", dir=root) as temporary:
        stage = Path(temporary)
        for name in REQUIRED:
            if not name.startswith("rank/"):
                continue
            _, target, family, filename = name.split("/")
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = root / "rank" / target / filename if manifest["build_version"] == 11 else root / name
            shutil.copy2(source, destination)
        if manifest["build_version"] in (14, 15, 16):
            # Stage both renamed tables before promoting: the old edge filename becomes
            # the new vector filename, so in-place renaming would overwrite its source.
            sources = {
                "recall/user/user_als_u2u.csv": "recall/user/user_als_emb.csv",
                "recall/user/user_als_emb.csv": "recall/user/user_als_emb_vectors.csv",
            }
            if manifest["build_version"] == 16:
                sources = {}
            sources.update({f"recall/{target}/text_search.csv": f"recall/{target}/sparse.csv"
                            for target in ("item", "user")})
            for name in REQUIRED:
                source = sources.get(name, name)
                if name.startswith("recall/") and source in manifest["outputs"]:
                    (stage / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(root / source, stage / name)
            if manifest["build_version"] == 14:
                generate_text_embeddings(data_dir / "item.csv", stage / "recall/item")
        else:
            generate(data_dir / "item.csv", data_dir / "event.csv", stage / "recall",
                     user_file=data_dir / "user.csv")
        manifest["build_version"] = BUILD_VERSION
        manifest["layout_migration"] = "v11-v16 to v17; item/user sparse CSVs renamed to text_search; v16 artifact bytes preserved"
        old_outputs = list(manifest["outputs"])
        manifest["outputs"] = {name: sha256(stage / name) for name in REQUIRED}
        (stage / "default.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if not valid(data_dir, stage):
            raise ValueError("migrated bundle failed validation")
        for name in (*REQUIRED, "default.manifest.json"):
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, root / name)
        for name in old_outputs:
            if name not in REQUIRED:
                (root / name).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-root", required=True)
    args = parser.parse_args()
    migrate(args.data, args.model_root)
