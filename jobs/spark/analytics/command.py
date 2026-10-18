"""Build the analytics Spark submission without importing PySpark."""

import os
import re

from jobs.spark.common.submission import spark_submit


def analytics_command(payload):
    date_from, date_to = payload.get("date_from"), payload.get("date_to")
    scene = payload.get("scene", "")
    for value in (date_from, date_to):
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", value or ""):
            raise ValueError("dates must use YYYY-MM-DD")
    if date_from > date_to:
        raise ValueError("date_from must not be after date_to")
    if scene and not re.match(r"^[A-Za-z0-9_-]+$", scene):
        raise ValueError("invalid scene")
    return spark_submit("analytics", "analytics/job.py") + [
        "--date-from",
        date_from,
        "--date-to",
        date_to,
        "--event-path",
        os.environ.get(
            "OPENREC_EVENT_PATH", "hdfs://namenode:8020/openrec/hive/event"
        ),
    ] + (["--scene", scene] if scene else [])
