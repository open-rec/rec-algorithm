"""Build the recall Spark submission without importing PySpark."""

import os
import re

from jobs.spark.common.submission import spark_submit
from .catalog import ALGORITHMS, PUBLISHABLE_ALGORITHMS, get_recall_spec


def recall_command(payload):
    algorithm = payload.get("algorithm")
    business_date = payload.get("date")
    revision = payload.get("revision", "r001")
    if algorithm not in ALGORITHMS:
        raise ValueError("unsupported recall algorithm: %s" % algorithm)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", business_date or ""):
        raise ValueError("date must use YYYY-MM-DD")
    if not re.match(r"^r\d{3,}$", revision):
        raise ValueError("revision must look like r001")
    target_type = payload.get("target_type", "item")
    if target_type not in ("item", "user"):
        raise ValueError("target_type must be item or user")
    output_name = "user_text_search" if algorithm == "text_search" and target_type == "user" else algorithm
    output_table = payload.get("output_table", "openrec.recall_%s" % output_name)
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_.]*$", output_table):
        raise ValueError("invalid output table")
    command = spark_submit("recall", "recall/job.py") + [
        algorithm,
        "--date",
        business_date,
        "--revision",
        revision,
        "--event-path",
        os.environ.get(
            "OPENREC_EVENT_PATH", "hdfs://namenode:8020/openrec/hive/event"
        ),
        "--item-path",
        os.environ.get(
            "OPENREC_ITEM_PATH", "hdfs://namenode:8020/openrec/hive/item"
        ),
        "--user-path",
        os.environ.get(
            "OPENREC_USER_PATH", "hdfs://namenode:8020/openrec/hive/user"
        ),
        "--output-path",
        os.environ.get(
            "OPENREC_RECALL_PATH", "hdfs://namenode:8020/openrec/hive/recall"
        )
        + "/"
        + output_name,
        "--es-host",
        "https://elasticsearch:9200",
        "--es-user",
        "elastic",
        "--es-password",
        os.environ.get("ELASTIC_PASSWORD", "openrec-es-password"),
        "--console-url",
        os.environ.get("REC_CONSOLE_URL", "http://rec-console:8095"),
        "--max-index-versions",
        str(payload.get("max_index_versions", 2)),
    ]
    if algorithm in PUBLISHABLE_ALGORITHMS:
        command.append("--publish")
    spec = get_recall_spec(algorithm)
    for name, default in spec.command_options:
        command.extend(["--" + name.replace("_", "-"), str(payload.get(name, default))])
    command.extend(["--size", str(payload.get("size", spec.default_size))])
    if algorithm == "text_search":
        command.extend(["--target-type", target_type])
        defaults = ["name", "gender", "country", "city", "tags"] if target_type == "user" else ["title", "category", "tags"]
        columns = payload.get("text_columns", defaults)
        if not isinstance(columns, list) or not columns or not all(
                isinstance(value, str) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value)
                for value in columns):
            raise ValueError("text_columns must be a non-empty list of column names")
        command.extend(["--text-columns", ",".join(columns)])
    return command
