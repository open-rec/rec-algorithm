"""Request, candidate and session features shared by training and serving."""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def materialize_request_context(context=None, candidate_ids=None, request_time=None,
                                candidate_contexts=None):
    """Return one context row per candidate with deterministic derived fields."""
    context = dict(context or {})
    candidate_ids = list(candidate_ids or [])
    candidate_contexts = candidate_contexts or {}
    timestamp = request_time if request_time is not None else context.get("request_time", 0)
    try:
        instant = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        instant = datetime.fromtimestamp(0, tz=timezone.utc)
    count = len(candidate_ids)
    rows = []
    for position, candidate_id in enumerate(candidate_ids):
        row = dict(context)
        row.update(candidate_contexts.get(str(candidate_id), {}) or {})
        row.setdefault("candidate_position", position)
        row.setdefault("candidate_count", count)
        row.setdefault("position_ratio", position / max(1, count - 1))
        row.setdefault("request_hour_sin", math.sin(2 * math.pi * instant.hour / 24))
        row.setdefault("request_hour_cos", math.cos(2 * math.pi * instant.hour / 24))
        row.setdefault("request_weekday_sin",
                       math.sin(2 * math.pi * instant.weekday() / 7))
        row.setdefault("request_weekday_cos",
                       math.cos(2 * math.pi * instant.weekday() / 7))
        rows.append(row)
    return pd.DataFrame(rows)


def materialize_session_features(events, as_of_time, session_id=None):
    """Build a point-in-time session row from events strictly before a label."""
    frame = events.copy() if events is not None else pd.DataFrame()
    if "time" not in frame:
        frame["time"] = pd.Series(dtype=float)
    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame = frame[frame["time"].notna() & (frame["time"] < as_of_time)]
    if session_id is not None and "session_id" in frame:
        frame = frame[frame["session_id"].fillna("").astype(str) == str(session_id)]
    types = frame.get("type", pd.Series("", index=frame.index)).fillna("").astype(str)
    values = pd.to_numeric(frame.get("value", pd.Series(0, index=frame.index)),
                           errors="coerce").fillna(0)
    last = float(frame["time"].max()) if not frame.empty else 0.0
    result = {
        "id": "" if session_id is None else str(session_id),
        "event_count": float(len(frame)),
        "event_value_sum": float(values.sum()),
        "event_value_mean": float(values.mean()) if len(frame) else 0.0,
        "event_unique_item_count": float(frame.get(
            "item_id", pd.Series(dtype=object)).nunique()),
        "event_first_time": float(frame["time"].min()) if len(frame) else 0.0,
        "event_last_time": last,
        "event_recency_seconds": max(0.0, float(as_of_time) - last) if last else 0.0,
    }
    for event_type in ("click", "expose", "buy", "collect", "stay"):
        result["event_%s_count" % event_type] = float((types == event_type).sum())
    return result


def candidate_interactions(events, candidate_ids, as_of_time):
    """Build generic user-candidate history signals without dataset assumptions."""
    frame = events.copy() if events is not None else pd.DataFrame()
    frame["time"] = pd.to_numeric(frame.get("time"), errors="coerce")
    frame = frame[frame["time"].notna() & (frame["time"] < as_of_time)]
    clicks = frame[frame.get("type", pd.Series("", index=frame.index)) == "click"]
    rows = []
    recent_items = (frame["item_id"].fillna("").astype(str)
                    if "item_id" in frame else pd.Series(dtype=str))
    for candidate_id in candidate_ids:
        selected = clicks[clicks.get("item_id", pd.Series("", index=clicks.index)).astype(str)
                          == str(candidate_id)]
        rows.append({
            "user_item_click_count_log": float(np.log1p(len(selected))),
            "candidate_seen_previous_1": float(str(candidate_id) in
                                                recent_items.tail(1).tolist()),
            "candidate_seen_previous_5": float(str(candidate_id) in
                                                recent_items.tail(5).tolist()),
        })
    return pd.DataFrame(rows)
