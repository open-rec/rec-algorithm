"""Distributed events recall formulas."""

from pyspark.sql import Window, functions as F
from algorithm.recall.recall import EVENT_UNIQUE_COLUMNS


def _events(events, event_type="click"):
    columns = [column for column in EVENT_UNIQUE_COLUMNS if column in events.columns]
    if "scene" in events.columns and "scene" not in columns:
        columns.append("scene")
    frame = events.dropDuplicates(columns).filter(
        F.col("scene").isNotNull() & F.col("item_id").isNotNull())
    return frame.filter(F.col("type") == event_type) if event_type else frame
