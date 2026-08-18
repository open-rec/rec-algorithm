"""Entity-level behavioural features derived from the event stream.

The result is deliberately a flat, stable table: it can be written to the offline user/item
feature table and copied to Redis without requiring rank-engine to read the event stream.
"""

import numpy as np
import pandas as pd


DEFAULT_EVENT_TYPES = ("click", "expose", "buy", "collect", "stay")
WINDOW_DAYS = (1, 7, 30)
DAY_SECONDS = 24 * 60 * 60


def event_feature_columns(counterpart_name, event_types=DEFAULT_EVENT_TYPES):
    columns = [
        "event_count", "event_value_sum", "event_value_mean", "event_active_days",
        "event_unique_scene_count", f"event_unique_{counterpart_name}_count",
        "event_first_time", "event_last_time", "event_recency_seconds",
    ]
    columns.extend(f"event_count_{days}d" for days in WINDOW_DAYS)
    columns.extend(f"event_{event_type}_count" for event_type in event_types)
    columns.append("event_click_rate")
    return columns


def aggregate_event_features(events, entity="user", as_of_time=None,
                             event_types=DEFAULT_EVENT_TYPES):
    """Aggregate events by user or item, returning one row per observed entity.

    ``as_of_time`` makes snapshots reproducible and prevents events after the snapshot from leaking
    in. When omitted, the newest valid event timestamp is used.
    """
    if entity not in ("user", "item"):
        raise ValueError("entity must be 'user' or 'item'")
    key = f"{entity}_id"
    counterpart = "item" if entity == "user" else "user"
    columns = [key] + event_feature_columns(counterpart, event_types)
    if events is None or events.empty or key not in events.columns:
        return pd.DataFrame(columns=columns)

    frame = events.copy()
    raw_time = frame["time"] if "time" in frame else pd.Series(np.nan, index=frame.index)
    frame["time"] = pd.to_numeric(raw_time, errors="coerce")
    frame = frame[frame[key].notna() & frame["time"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    snapshot = float(frame["time"].max()) if as_of_time is None else float(as_of_time)
    frame = frame[frame["time"] <= snapshot].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    raw_value = frame["value"] if "value" in frame else pd.Series(0.0, index=frame.index)
    frame["value_num"] = pd.to_numeric(raw_value, errors="coerce").fillna(0.0)
    if "scene" not in frame:
        frame["scene"] = ""
    frame["event_day"] = np.floor(frame["time"] / DAY_SECONDS)
    grouped = frame.groupby(key, sort=False)
    result = grouped.agg(
        event_count=(key, "size"),
        event_value_sum=("value_num", "sum"),
        event_value_mean=("value_num", "mean"),
        event_active_days=("event_day", "nunique"),
        event_unique_scene_count=("scene", "nunique"),
        event_first_time=("time", "min"),
        event_last_time=("time", "max"),
    )
    counterpart_key = f"{counterpart}_id"
    if counterpart_key in frame.columns:
        result[f"event_unique_{counterpart}_count"] = grouped[counterpart_key].nunique()
    else:
        result[f"event_unique_{counterpart}_count"] = 0
    result["event_recency_seconds"] = snapshot - result["event_last_time"]

    for days in WINDOW_DAYS:
        recent = frame[frame["time"] >= snapshot - days * DAY_SECONDS]
        result[f"event_count_{days}d"] = recent.groupby(key).size().reindex(result.index, fill_value=0)

    event_type = frame.get("type", pd.Series("", index=frame.index)).fillna("").astype(str)
    for name in event_types:
        counts = frame[event_type == name].groupby(key).size()
        result[f"event_{name}_count"] = counts.reindex(result.index, fill_value=0)
    click_count = result.get("event_click_count", pd.Series(0, index=result.index))
    expose_count = result.get("event_expose_count", pd.Series(0, index=result.index))
    denominator = click_count + expose_count
    result["event_click_rate"] = np.where(
        denominator > 0, click_count / denominator, 0.0)
    return result.reset_index()[columns]


def enrich_entity_features(entities, events=None, entity="user", as_of_time=None,
                           event_types=DEFAULT_EVENT_TYPES):
    """Left join behavioural columns onto a user/item table, zero-filling unseen entities."""
    if entities is None:
        return None
    result = entities.copy()
    counterpart = "item" if entity == "user" else "user"
    feature_columns = event_feature_columns(counterpart, event_types)
    if events is not None:
        aggregated = aggregate_event_features(events, entity, as_of_time, event_types)
        # Recomputing a snapshot replaces stale columns rather than creating _x/_y duplicates.
        result = result.drop(columns=[c for c in feature_columns if c in result.columns])
        result = result.merge(aggregated, how="left", left_on="id", right_on=f"{entity}_id")
        result = result.drop(columns=[f"{entity}_id"], errors="ignore")
    for column in feature_columns:
        if column not in result.columns:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result
