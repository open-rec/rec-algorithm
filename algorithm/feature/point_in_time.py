"""Point-in-time feature materialization for rank training samples."""

from bisect import bisect_right

import pandas as pd

from algorithm.feature.event_feature import enrich_entity_features


def resolve_event_mutations_as_of(events, observation_cutoff):
    """Resolve one reproducible event population at a fixed mutation observation time."""
    if events is None or events.empty:
        return pd.DataFrame(columns=events.columns if events is not None else None)
    frame = events.copy()
    event_id = frame.get("event_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    trace = frame.get("trace_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    fallback = (frame.get("user_id", pd.Series("", index=frame.index)).astype(str) + "|" +
                frame.get("item_id", pd.Series("", index=frame.index)).astype(str) + "|" +
                frame.get("scene", pd.Series("", index=frame.index)).astype(str) + "|" +
                frame.get("type", pd.Series("", index=frame.index)).astype(str) + "|" +
                frame.get("time", pd.Series("", index=frame.index)).astype(str) + "|" + trace)
    frame["_event_key"] = event_id.where(event_id.str.strip().ne(""), fallback)
    effective = frame.get("_effective_time", frame.get(
        "occurred_at", pd.Series(0, index=frame.index)))
    frame["_effective_time"] = pd.to_numeric(effective, errors="coerce").fillna(0)
    operation = frame.get("_operation", frame.get(
        "operation", pd.Series("INSERT", index=frame.index))).fillna("INSERT").astype(str).str.upper()
    frame["_operation"] = operation
    frame["_delete_order"] = operation.eq("DELETE").astype(int)
    visible = frame[frame["_effective_time"] <= observation_cutoff].sort_values(
        ["_effective_time", "_delete_order"], kind="mergesort") \
        .drop_duplicates("_event_key", keep="last")
    return visible[visible["_operation"] != "DELETE"].drop(
        columns=["_event_key", "_effective_time", "_delete_order", "_operation"], errors="ignore")


def _history_index(frame, fallback_times):
    result = {}
    if frame is None or frame.empty:
        return result
    rows = frame.copy()
    effective = rows.get("_effective_time")
    if effective is None:
        effective = pd.Series(0, index=rows.index)
        for name in fallback_times:
            if name in rows:
                effective = effective.where(effective.ne(0), pd.to_numeric(
                    rows[name], errors="coerce").fillna(0))
    rows["_effective_time"] = pd.to_numeric(effective, errors="coerce").fillna(0).astype("int64")
    for entity_id, group in rows[rows["id"].notna()].groupby("id", sort=False):
        operations = group.get("_operation", pd.Series("INSERT", index=group.index))
        group["_delete_order"] = operations.astype(str) \
            .str.upper().eq("DELETE").astype(int)
        ordered = group.sort_values(["_effective_time", "_delete_order"], kind="mergesort")
        result[entity_id] = (ordered["_effective_time"].tolist(), ordered)
    return result


def _as_of(index, entity_id, label_time):
    history = index.get(entity_id)
    if history is None:
        return None
    times, rows = history
    position = bisect_right(times, label_time) - 1
    if position < 0:
        return None
    row = rows.iloc[position]
    if str(row.get("_operation", "INSERT")).upper() == "DELETE":
        return None
    return row.drop(labels=["_effective_time", "_delete_order", "_operation",
                            "_mutation_time", "dt"],
                    errors="ignore").to_dict()


def _events_by_entity(events, key):
    result = {}
    if events is None or events.empty or key not in events:
        return result
    frame = events.copy()
    frame["time"] = pd.to_numeric(frame.get("time"), errors="coerce")
    frame["_effective_time"] = pd.to_numeric(
        frame.get("_effective_time", frame["time"]), errors="coerce").fillna(frame["time"])
    frame = frame[frame[key].notna() & frame["time"].notna()]
    event_id = frame.get("event_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    trace_id = frame.get("trace_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    fallback = (frame.get("user_id", "").astype(str) + "|" +
                frame.get("item_id", "").astype(str) + "|" +
                frame.get("scene", "").astype(str) + "|" +
                frame.get("type", "").astype(str) + "|" + frame["time"].astype(str))
    traced = fallback + "|" + trace_id
    frame["_event_key"] = event_id.where(event_id.str.len() > 0,
                                          traced.where(trace_id.str.len() > 0, fallback))
    operations = frame.get("_operation", pd.Series("INSERT", index=frame.index))
    frame["_delete_order"] = operations.astype(str) \
        .str.upper().eq("DELETE").astype(int)
    frame = frame.sort_values(["_effective_time", "_delete_order"], kind="mergesort")
    for entity_id, group in frame.groupby(key, sort=False):
        result[entity_id] = (group["_effective_time"].tolist(), group)
    return result


def _behavior_as_of(index, entity_id, label_time):
    history = index.get(entity_id)
    if history is None:
        return pd.DataFrame()
    times, rows = history
    # Mutations are visible at their effective time, but only interactions strictly before the
    # label contribute. Resolve each event's latest mutation first so future deletes cannot rewrite
    # an earlier sample.
    position = bisect_right(times, label_time)
    visible = rows.iloc[:position].drop_duplicates("_event_key", keep="last")
    operations = visible.get("_operation", pd.Series("INSERT", index=visible.index))
    visible = visible[operations.astype(str).str.upper() != "DELETE"]
    return visible[visible["time"] < label_time].drop(
        columns=["_effective_time", "_event_key", "_delete_order", "_operation",
                 "_mutation_time", "dt"], errors="ignore")


def materialize_point_in_time_samples(events, feature_events, users, items,
                                      target_type="item"):
    """Return labels and aligned user/candidate feature rows as they existed at label time."""
    if target_type not in ("item", "user"):
        raise ValueError("target_type must be item or user")
    user_history = _history_index(users, ("login_time", "register_time"))
    item_history = user_history if target_type == "user" else _history_index(
        items, ("modify_time", "pub_time"))
    user_events = _events_by_entity(feature_events, "user_id")
    item_events = user_events if target_type == "user" else _events_by_entity(
        feature_events, "item_id")
    labels, sample_users, sample_items = [], [], []
    labels_frame = events[events["type"].isin(("click", "expose"))]
    for _, event in labels_frame.sort_values("time", kind="mergesort").iterrows():
        label_time = int(event["time"])
        user = _as_of(user_history, event["user_id"], label_time)
        item = _as_of(item_history, event["item_id"], label_time)
        if user is None or item is None:
            continue
        user_frame = enrich_entity_features(pd.DataFrame([user]), _behavior_as_of(
            user_events, event["user_id"], label_time), "user", label_time)
        item_entity = "user" if target_type == "user" else "item"
        item_frame = enrich_entity_features(pd.DataFrame([item]), _behavior_as_of(
            item_events, event["item_id"], label_time), item_entity, label_time)
        labels.append(event.to_dict())
        sample_users.append(user_frame.iloc[0].to_dict())
        sample_items.append(item_frame.iloc[0].to_dict())
    return (pd.DataFrame(labels).reset_index(drop=True),
            pd.DataFrame(sample_users).reset_index(drop=True),
            pd.DataFrame(sample_items).reset_index(drop=True))
