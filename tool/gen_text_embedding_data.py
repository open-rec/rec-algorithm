"""Generate item language vectors; queries use the same TextEmbeddingEncoder."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from algorithm.recall.text_embedding import TextEmbeddingEncoder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(item_file, output_dir, encoder=None, batch_size=32):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    encoder = encoder or TextEmbeddingEncoder()
    items = pd.read_csv(item_file, dtype=str).fillna("")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tables = {"item_title_emb": ["title"], "item_content_emb": ["title", "category", "tags"]}
    for name, columns in tables.items():
        texts = items[columns].agg(" ".join, axis=1).tolist()
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["scene", "item", "vector"])
            for start in range(0, len(items), batch_size):
                vectors = encoder.encode(texts[start:start + batch_size], batch_size=batch_size)
                if start % (batch_size * 20) == 0:
                    print(f"{name}: {start}/{len(items)}", flush=True)
                rows = items.iloc[start:start + batch_size]
                for row, vector in zip(rows.itertuples(index=False), vectors):
                    writer.writerow([row.scene, row.id, vector.tolist()])
        print(f"Generated {name}: {len(items)} rows", flush=True)
    metadata = {"encoder": encoder.metadata(), "fields": tables,
                "input_sha256": digest(item_file),
                "outputs": {name + ".csv": digest(output / (name + ".csv")) for name in tables}}
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="model ID; otherwise OPENREC_TEXT_EMBEDDING_MODEL")
    parser.add_argument("--revision", help="model revision; otherwise OPENREC_TEXT_EMBEDDING_REVISION")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    metadata = generate(args.item, args.output, TextEmbeddingEncoder(args.model, args.revision), args.batch_size)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
