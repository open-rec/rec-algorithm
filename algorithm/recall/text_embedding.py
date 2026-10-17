"""Shared Qwen3 query/document encoder configured by model ID and environment."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TextEmbeddingEncoder:
    def __init__(self, model_name=None, revision=None, cache_dir=None, device=None):
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name or os.getenv(
            "OPENREC_TEXT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        if self.model_name.startswith(("/", ".")) or ".." in self.model_name.split("/"):
            raise ValueError("Configure a model ID, not a filesystem path")
        self.revision = revision or os.getenv("OPENREC_TEXT_EMBEDDING_REVISION", "main")
        self.max_length = int(os.getenv("OPENREC_TEXT_EMBEDDING_MAX_LENGTH", "512"))
        self.instruction = os.getenv("OPENREC_TEXT_EMBEDDING_INSTRUCTION",
                                     "Given a search query, retrieve relevant items that match the query")
        self.device = device or os.getenv("OPENREC_TEXT_EMBEDDING_DEVICE", "cpu")
        cache = cache_dir or os.getenv("OPENREC_MODEL_CACHE")
        local = Path(cache) / self.model_name if cache else None
        self.model_files = {}
        source = self.model_name
        options = dict(revision=self.revision, cache_dir=cache, trust_remote_code=False)
        # ModelScope exports use <cache>/<namespace>/<model>; callers still supply a model ID.
        if local and (local / "config.json").is_file():
            files = sorted(path for path in local.iterdir()
                           if path.suffix in (".json", ".safetensors", ".txt") and path.is_file())
            self.model_files = {path.name: file_digest(path) for path in files}
            fingerprint = "sha256:" + hashlib.sha256(
                json.dumps(self.model_files, sort_keys=True).encode()).hexdigest()
            if self.revision not in ("main", fingerprint):
                raise ValueError("Local model revision does not match the configured fingerprint")
            self.revision = fingerprint
            source = str(local)
            options = dict(local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(source, padding_side="left", **options)
        self.model = AutoModel.from_pretrained(source, **options).to(self.device).eval()
        if self.model.config.model_type != "qwen3":
            raise ValueError("This encoder requires Qwen3-Embedding; other families need their own recipe")

    def encode(self, texts, *, query=False, batch_size=32):
        import torch
        import torch.nn.functional as functional

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        prefix = f"Instruct: {self.instruction}\nQuery:" if query else ""
        batches = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                inputs = self.tokenizer([prefix + text for text in texts[start:start + batch_size]],
                                        max_length=self.max_length, padding=True,
                                        truncation=True, return_tensors="pt").to(self.device)
                hidden = self.model(**inputs, use_cache=False).last_hidden_state
                # Left padding keeps the final valid token at the end for every document.
                pooled = hidden[:, -1]
                batches.append(functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, self.model.config.hidden_size))

    def encode_query(self, text):
        return self.encode([text], query=True)[0]

    def metadata(self):
        return {"model_name": self.model_name,
                "revision": self.revision if self.model_files else (
                    getattr(self.model.config, "_commit_hash", None) or self.revision),
                "model_files_sha256": self.model_files,
                "dimension": self.model.config.hidden_size, "pooling": "last_valid_token",
                "normalization": "l2", "query_prefix": f"Instruct: {self.instruction}\nQuery:",
                "document_prefix": "", "padding_side": "left", "max_length": self.max_length}
