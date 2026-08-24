import pytest

from jobs.spark.runner import analytics_command, rank_command, recall_command


def test_runner_builds_versioned_es_publish_command(monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "secret")
    command = recall_command({
        "algorithm": "i2i",
        "date": "2026-08-19",
        "revision": "r002",
        "max_index_versions": 2,
    })
    assert command[:7] == [
        "/opt/spark/bin/spark-submit", "--master", "spark://spark-master:7077",
        "--deploy-mode", "client", "--total-executor-cores", "4",
    ]
    assert command[command.index("--date") + 1] == "2026-08-19"
    assert command[command.index("--revision") + 1] == "r002"
    assert command[command.index("--es-password") + 1] == "secret"
    assert command[command.index("--console-url") + 1] == "http://rec-console:8095"
    assert command[command.index("--event-path") + 1].endswith("/openrec/hive/event")
    assert command[command.index("--item-path") + 1].endswith("/openrec/hive/item")
    assert command[command.index("--output-path") + 1].endswith("/openrec/hive/recall/i2i")
    assert "--publish" in command


@pytest.mark.parametrize("payload", [
    {"algorithm": "unknown", "date": "2026-08-19"},
    {"algorithm": "hot", "date": "20260819"},
    {"algorithm": "new", "date": "2026-08-19", "revision": "1"},
])
def test_runner_rejects_invalid_job(payload):
    with pytest.raises(ValueError):
        recall_command(payload)


def test_rank_command_caps_spark_and_uses_cumulative_entity_paths():
    command = rank_command({"date": "2026-08-21", "revision": "r002",
                            "scene": "scene_0", "epochs": 3, "min_auc": .5,
                            "model_type": "fm", "factor_dim": 16})
    assert command[command.index("--total-executor-cores") + 1] == "4"
    assert command[command.index("--user-path") + 1].endswith("/openrec/hive/user")
    assert command[command.index("--artifact-root") + 1] == "/models/releases"
    assert command[command.index("--min-auc") + 1] == "0.5"
    assert command[command.index("--model-type") + 1] == "fm"
    assert command[command.index("--factor-dim") + 1] == "16"


@pytest.mark.parametrize("payload", [
    {"date": "2026-08-21", "model_type": "deepfm"},
    {"date": "2026-08-21", "model_type": "fm", "factor_dim": 0},
])
def test_rank_command_rejects_invalid_model_options(payload):
    with pytest.raises(ValueError):
        rank_command(payload)


def test_analytics_command_caps_spark_and_filters_range():
    command = analytics_command({"date_from": "2026-08-14", "date_to": "2026-08-21",
                                 "scene": "scene_0"})
    assert command[command.index("--total-executor-cores") + 1] == "4"
    assert command[command.index("--date-from") + 1] == "2026-08-14"
    assert command[command.index("--date-to") + 1] == "2026-08-21"
    assert command[command.index("--scene") + 1] == "scene_0"
