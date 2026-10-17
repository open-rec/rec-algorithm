import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np
import torch

from algorithm.recall.text_embedding import TextEmbeddingEncoder


def test_shared_model_name_cache_and_last_token_pooling(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENREC_TEXT_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setenv("OPENREC_MODEL_CACHE", "/configured-cache")
    monkeypatch.setenv("OPENREC_TEXT_EMBEDDING_REVISION", "pinned")
    model = Mock()
    model.to.return_value.eval.return_value = model
    model.config = SimpleNamespace(model_type="qwen3", hidden_size=2, _commit_hash="pinned")
    model.return_value = SimpleNamespace(last_hidden_state=torch.tensor([[[900., 900.], [3., 4.]]]))
    class Batch(dict):
        def to(self, device):
            return self
    tokenizer = Mock(return_value=Batch(attention_mask=torch.tensor([[0, 1]])))
    auto_model = Mock(from_pretrained=Mock(return_value=model))
    auto_tokenizer = Mock(from_pretrained=Mock(return_value=tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=auto_model, AutoTokenizer=auto_tokenizer))
    encoder = TextEmbeddingEncoder()
    assert np.allclose(encoder.encode_query("hello"), [.6, .8])
    assert tokenizer.call_args.args[0] == ["Instruct: Given a search query, retrieve relevant items that match the query\nQuery:hello"]
    assert np.allclose(encoder.encode(["hello"]), [[.6, .8]])
    assert tokenizer.call_args.args[0] == ["hello"]
    assert auto_model.from_pretrained.call_args.args == ("Qwen/Qwen3-Embedding-0.6B",)
    assert auto_model.from_pretrained.call_args.kwargs["cache_dir"] == "/configured-cache"
    assert auto_tokenizer.from_pretrained.call_args.kwargs["revision"] == "pinned"
    assert "/configured-cache" not in str(encoder.metadata())

    local = tmp_path / "Qwen/Qwen3-Embedding-0.6B"
    local.mkdir(parents=True)
    (local / "config.json").write_text('{}')
    (local / "model.safetensors").write_bytes(b"test weights")
    local_encoder = TextEmbeddingEncoder(cache_dir=str(tmp_path), revision="main")
    fingerprint = local_encoder.metadata()["revision"]
    assert fingerprint.startswith("sha256:")
    assert local_encoder.metadata()["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert auto_model.from_pretrained.call_args.kwargs["local_files_only"] is True
    assert str(tmp_path) not in str(local_encoder.metadata())
    TextEmbeddingEncoder(cache_dir=str(tmp_path), revision=fingerprint)
    (local / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="revision"):
        TextEmbeddingEncoder(cache_dir=str(tmp_path), revision=fingerprint)
