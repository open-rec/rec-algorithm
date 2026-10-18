"""Extension and gateway contracts that do not require PySpark."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from jobs.spark.common.session import get_spark
from jobs.spark.recall.catalog import RECALL_JOBS, RecallSpec, get_recall_spec
from jobs.spark.recall.command import recall_command
from jobs.spark.runner import job_result


def test_gateway_import_does_not_load_spark_or_models():
    subprocess.run([sys.executable, "-c", "import sys; import jobs.spark.runner; "
                    "assert not {'pyspark', 'torch', 'pandas'} & sys.modules.keys()"], check=True)


@pytest.mark.parametrize("name", RECALL_JOBS)
def test_all_registered_algorithms_can_be_submitted(name):
    spec = get_recall_spec(name)
    command = recall_command({"algorithm": name, "date": "2026-10-18"})
    assert name in command
    assert "--publish" in command
    assert command[command.index("--size") + 1] == str(spec.default_size)
    for option, default in spec.command_options:
        assert command[command.index("--" + option.replace("_", "-")) + 1] == str(default)


def test_extension_resolves_handler_without_pipeline_changes(monkeypatch):
    module = SimpleNamespace(compute=lambda inputs, args: (inputs, args.size))
    monkeypatch.setitem(sys.modules, "example_recall_plugin", module)
    spec = RecallSpec("example_recall_plugin:compute", "example")
    assert spec.compute("snapshot", SimpleNamespace(size=7)) == ("snapshot", 7)


def test_unknown_algorithm_does_not_fall_back_to_embedding():
    with pytest.raises(ValueError, match="unsupported recall algorithm"):
        get_recall_spec("unknown")


def test_injected_session_does_not_require_pyspark():
    spark = object()
    assert get_spark("test", spark, hive=True) is spark


@pytest.mark.parametrize("path,marker,key", [
    ("/jobs/rank/train", "OPENREC_MODEL_MANIFEST=", "manifest"),
    ("/jobs/analytics", "OPENREC_ANALYTICS=", "analytics"),
])
def test_gateway_extracts_last_result_and_rejects_missing(path, marker, key):
    output = marker + '{}\nlogs\n' + marker + json.dumps({"version": 2})
    assert job_result(path, output)[key] == {"version": 2}
    with pytest.raises(RuntimeError, match="missing"):
        job_result(path, "logs only")
    with pytest.raises(ValueError):
        job_result(path, marker + "broken json")
