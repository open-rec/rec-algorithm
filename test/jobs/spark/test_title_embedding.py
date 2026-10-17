import io
import json
from unittest.mock import patch

import pytest

from jobs.spark.title_embedding import embed_titles
from jobs.spark.runner import recall_command


def response(data, model="Qwen/Qwen3-Embedding-0.6B"):
    return io.BytesIO(json.dumps({"model": model, "data": data}).encode())


def test_raw_titles_order_and_normalization():
    with patch("urllib.request.urlopen", return_value=response([
            {"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 4]}])) as call:
        vectors = embed_titles(["title one", "title two"], "http://vllm/v1/embeddings",
                               "Qwen/Qwen3-Embedding-0.6B", 2, "test-key")
    assert vectors == [[0.6, 0.8], [0, 1]]
    request = call.call_args.args[0]
    assert json.loads(request.data)["input"] == ["title one", "title two"]
    assert request.get_header("Authorization") == "Bearer test-key"


@pytest.mark.parametrize("data", [[], [{"index": 1, "embedding": [1, 0]}],
    [{"index": 0, "embedding": [0, 0]}], [{"index": 0, "embedding": [1]}],
    [{"index": 0, "embedding": [float("nan"), 0]}]])
def test_invalid_response_rejects_entire_batch(data):
    with patch("urllib.request.urlopen", return_value=response(data)), pytest.raises(ValueError):
        embed_titles(["title"], "http://vllm/v1/embeddings", "Qwen/Qwen3-Embedding-0.6B", 2)


def test_model_mismatch_rejected():
    with patch("urllib.request.urlopen", return_value=response([], model="other")), pytest.raises(ValueError):
        embed_titles(["title"], "http://vllm/v1/embeddings", "Qwen/Qwen3-Embedding-0.6B", 2)


def test_runner_routes_title_job_to_output_and_publisher():
    command = recall_command({"algorithm": "query_title_emb", "date": "2026-10-17"})
    assert "query_title_emb" in command and "--publish" in command
    assert command[command.index("--output-path") + 1].endswith("/query_title_emb")
