import pytest

from jobs.spark.runner import analytics_command, rank_command, recall_command


def test_runner_builds_versioned_es_publish_command(monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "secret")
    command = recall_command(
        {
            "algorithm": "item_cf_i2i",
            "date": "2026-08-19",
            "revision": "r002",
            "max_index_versions": 2,
        }
    )
    assert command[:7] == [
        "/opt/spark/bin/spark-submit",
        "--master",
        "spark://spark-master:7077",
        "--deploy-mode",
        "client",
        "--total-executor-cores",
        "4",
    ]
    assert command[command.index("--date") + 1] == "2026-08-19"
    assert command[command.index("--revision") + 1] == "r002"
    assert command[command.index("--es-password") + 1] == "secret"
    assert (
        command[command.index("--console-url") + 1]
        == "http://rec-console:8095"
    )
    assert command[command.index("--event-path") + 1].endswith(
        "/openrec/hive/event"
    )
    assert command[command.index("--item-path") + 1].endswith(
        "/openrec/hive/item"
    )
    assert command[command.index("--output-path") + 1].endswith(
        "/openrec/hive/recall/item_cf_i2i"
    )
    assert "--publish" in command


@pytest.mark.parametrize(
    "algorithm",
    [
        "user_cf_u2i",
        "content_i2i",
        "user_cf_u2u",
        "content_u2u",
        "user_emb_u2u",
    ],
)
def test_runner_builds_publishable_complementary_recall_jobs(algorithm):
    command = recall_command({"algorithm": algorithm, "date": "2026-08-19"})

    assert command[command.index("--output-path") + 1].endswith(
        "/recall/" + algorithm
    )
    assert command[command.index("--user-path") + 1].endswith(
        "/openrec/hive/user"
    )
    assert "--publish" in command


def test_runner_builds_sparse_document_job_with_selected_text_columns():
    command = recall_command({
        "algorithm": "sparse", "date": "2026-08-19",
        "text_columns": ["title", "brand", "tags"],
    })

    assert command[command.index("--text-columns") + 1] == "title,brand,tags"
    assert command[command.index("--output-path") + 1].endswith("/recall/sparse")
    assert "--publish" in command


@pytest.mark.parametrize(
    "payload",
    [
        {"algorithm": "unknown", "date": "2026-08-19"},
        {"algorithm": "hot", "date": "20260819"},
        {"algorithm": "new", "date": "2026-08-19", "revision": "1"},
    ],
)
def test_runner_rejects_invalid_job(payload):
    with pytest.raises(ValueError):
        recall_command(payload)


def test_rank_command_caps_spark_and_uses_cumulative_entity_paths():
    command = rank_command(
        {
            "date": "2026-08-21",
            "revision": "r002",
            "scene": "scene_0",
            "epochs": 3,
            "min_auc": 0.5,
            "model_type": "fm",
            "factor_dim": 16,
            "user_label_window_days": 14,
        }
    )
    assert command[command.index("--total-executor-cores") + 1] == "4"
    assert command[command.index("--user-path") + 1].endswith(
        "/openrec/hive/user"
    )
    assert command[command.index("--artifact-root") + 1] == "/models/releases"
    assert command[command.index("--min-auc") + 1] == "0.5"
    assert command[command.index("--model-type") + 1] == "fm"
    assert command[command.index("--factor-dim") + 1] == "16"
    assert command[command.index("--user-label-window-days") + 1] == "14"
    assert command[command.index("--max-history-rows") + 1] == "5000000"


def test_rank_command_accepts_lightgbm():
    command = rank_command(
        {"date": "2026-08-21", "model_type": "lightgbm"}
    )
    assert command[command.index("--model-type") + 1] == "lightgbm"


@pytest.mark.parametrize(
    "payload",
    [
        {"date": "2026-08-21", "model_type": "deepfm"},
        {"date": "2026-08-21", "model_type": "fm", "factor_dim": 0},
        {
            "date": "2026-08-21",
            "target_type": "user",
            "user_label_window_days": 31,
        },
        {"date": "2026-08-21", "max_events": 100, "max_history_rows": 99},
    ],
)
def test_rank_command_rejects_invalid_model_options(payload):
    with pytest.raises(ValueError):
        rank_command(payload)


def test_analytics_command_caps_spark_and_filters_range():
    command = analytics_command(
        {
            "date_from": "2026-08-14",
            "date_to": "2026-08-21",
            "scene": "scene_0",
        }
    )
    assert command[command.index("--total-executor-cores") + 1] == "4"
    assert command[command.index("--date-from") + 1] == "2026-08-14"
    assert command[command.index("--date-to") + 1] == "2026-08-21"
    assert command[command.index("--scene") + 1] == "scene_0"


def test_rank_command_preserves_feature_order_and_training_options():
    import json

    selection = {
        "user": ["user.age", "user.country"],
        "candidate": ["item.weight"],
    }
    command = rank_command(
        {
            "date": "2026-09-16",
            "scene": "global",
            "feature_selection": selection,
            "batch_size": 32,
            "validation_ratio": 0.3,
        }
    )
    assert (
        json.loads(command[command.index("--feature-selection") + 1])
        == selection
    )
    assert command[command.index("--batch-size") + 1] == "32"
    assert command[command.index("--validation-ratio") + 1] == "0.3"


def test_feature_gateway_is_available_without_inference_service():
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from jobs.spark.runner import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:%s" % httpd.server_port
    try:
        with urllib.request.urlopen(base + "/features") as response:
            assert "lr" in json.load(response)["data"]["models"]
        selection = {"user": ["user.age"], "candidate": ["item.weight"]}
        request = urllib.request.Request(
            base + "/features/validate",
            data=json.dumps({"feature_selection": selection}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["data"] == selection
        request.data = b'{"feature_selection": {}}'
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 422
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
