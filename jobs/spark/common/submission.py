"""Shared Spark submission defaults; safe to import without a Spark runtime."""

import os


def spark_submit(kind, script):
    return [
        "/opt/spark/bin/spark-submit",
        "--master", os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077"),
        "--deploy-mode", "client",
        "--total-executor-cores", os.environ.get(kind.upper() + "_SPARK_CORES", "4"),
        "--py-files", "/opt/openrec/rec-algorithm.zip",
        "/opt/openrec/jobs/spark/" + script,
    ]
