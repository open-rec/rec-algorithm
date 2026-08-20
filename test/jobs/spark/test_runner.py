import pytest

from jobs.spark.runner import recall_command


def test_runner_builds_versioned_es_publish_command(monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "secret")
    command = recall_command({
        "algorithm": "i2i",
        "date": "2026-08-19",
        "revision": "r002",
        "max_index_versions": 2,
    })
    assert command[:5] == [
        "/opt/spark/bin/spark-submit", "--master", "spark://spark-master:7077",
        "--deploy-mode", "client",
    ]
    assert command[command.index("--date") + 1] == "2026-08-19"
    assert command[command.index("--revision") + 1] == "r002"
    assert command[command.index("--es-password") + 1] == "secret"
    assert command[command.index("--console-url") + 1] == "http://rec-console:8095"
    assert "--publish" in command


@pytest.mark.parametrize("payload", [
    {"algorithm": "embedding", "date": "2026-08-19"},
    {"algorithm": "hot", "date": "20260819"},
    {"algorithm": "new", "date": "2026-08-19", "revision": "1"},
])
def test_runner_rejects_invalid_job(payload):
    with pytest.raises(ValueError):
        recall_command(payload)
