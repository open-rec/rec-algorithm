"""Point-in-time category preference and price sensitivity features."""

import json
from bisect import bisect_right

import numpy as np
import pandas as pd


WINDOWS = (("1d", 86400), ("7d", 604800), ("30d", 2592000))
PRICE_EVENT_TYPES = ("expose", "click", "collect", "buy")
CATEGORY_WEIGHTS = {
    "expose": 0.1,
    "click": 1.0,
    "stay": 1.0,
    "collect": 3.0,
    "buy": 5.0,
}


def commerce_feature_columns():
    columns = [
        "preferred_categories", "preferred_subcategories",
        "event_price_mean", "event_price_std",
    ]
    for event_type in PRICE_EVENT_TYPES:
        columns.extend((
            "event_%s_price_mean" % event_type,
            "event_%s_price_std" % event_type,
        ))
    for suffix, _ in WINDOWS:
        columns.extend((
            "event_click_price_mean_%s" % suffix,
            "event_buy_price_mean_%s" % suffix,
        ))
    columns.extend((
        "event_buy_to_click_price_ratio",
        "event_recent_to_long_click_price_ratio",
    ))
    return columns


def _json_object(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _price(row):
    for name in ("price", "unit_price", "unitPrice"):
        value = row.get(name)
        try:
            number = float(value)
            if np.isfinite(number) and number >= 0:
                return number
        except (TypeError, ValueError):
            pass
    ext = _json_object(row.get("ext_fields", row.get("extFields")))
    for name in ("price", "unitPrice", "unit_price"):
        try:
            number = float(ext.get(name))
            if np.isfinite(number) and number >= 0:
                return number
        except (TypeError, ValueError):
            pass
    return np.nan


def _item_history(items):
    result = {}
    if items is None or items.empty or "id" not in items:
        return result
    frame = items.copy()
    effective = frame.get("_effective_time")
    if effective is None:
        effective = frame.get("modify_time", frame.get("modifyTime"))
    if effective is None:
        effective = pd.Series(0, index=frame.index)
    frame["_effective_time"] = pd.to_numeric(effective, errors="coerce").fillna(0)
    operations = frame.get("_operation", pd.Series("INSERT", index=frame.index))
    frame["_delete_order"] = operations.fillna("INSERT").astype(str) \
        .str.upper().eq("DELETE").astype(int)
    for item_id, rows in frame[frame["id"].notna()].groupby("id", sort=False):
        ordered = rows.sort_values(
            ["_effective_time", "_delete_order"], kind="mergesort")
        result[item_id] = (ordered["_effective_time"].tolist(), ordered)
    return result


def enrich_events_with_item_context(events, items):
    """Attach the Item version visible at each event time."""
    if events is None or events.empty:
        return pd.DataFrame(columns=list(events.columns) if events is not None else None)
    history = _item_history(items)
    rows = []
    for _, event in events.iterrows():
        result = event.to_dict()
        event_ext = _json_object(event.get("ext_fields", event.get("extFields")))
        frozen = event_ext.get("_openrecItemContext", {})
        frozen = frozen if isinstance(frozen, dict) else {}
        event_time = pd.to_numeric(event.get("time"), errors="coerce")
        item_history = history.get(event.get("item_id"))
        item = None
        if item_history is not None and pd.notna(event_time):
            times, versions = item_history
            position = bisect_right(times, float(event_time)) - 1
            if position >= 0:
                candidate = versions.iloc[position]
                if str(candidate.get("_operation", "INSERT")).upper() != "DELETE":
                    item = candidate
        result["_item_category"] = str(frozen.get("category", "") or "") \
            if frozen else ("" if item is None else str(item.get("category", "") or ""))
        result["_item_subcategory"] = str(frozen.get("subcategory", "") or "") \
            if frozen else ("" if item is None else str(
                item.get("subcategory", item.get("subCategory", "")) or ""))
        result["_item_price"] = _price(frozen) if frozen else (
            np.nan if item is None else _price(item))
        rows.append(result)
    return pd.DataFrame(rows)


def _weighted_preferences(frame, field, limit):
    values = {}
    for _, event in frame.iterrows():
        name = str(event.get(field, "") or "").strip()
        if not name:
            continue
        weight = CATEGORY_WEIGHTS.get(str(event.get("type", "")).lower(), 0.0)
        raw_value = pd.to_numeric(event.get("value"), errors="coerce")
        if event.get("type") == "stay" and pd.notna(raw_value) and raw_value > 0:
            weight *= np.log1p(float(raw_value))
        values[name] = values.get(name, 0.0) + weight
    return ",".join(name for name, _ in sorted(
        values.items(), key=lambda value: (-value[1], value[0]))[:limit])


def aggregate_user_commerce_features(events, items, as_of_time=None, top_k=8):
    """Aggregate category and price features without using future Item versions."""
    columns = ["user_id"] + commerce_feature_columns()
    if events is None or events.empty or "user_id" not in events:
        return pd.DataFrame(columns=columns)
    frame = enrich_events_with_item_context(events, items)
    frame["time"] = pd.to_numeric(frame.get("time"), errors="coerce")
    frame = frame[frame["user_id"].notna() & frame["time"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    snapshot = float(frame["time"].max()) if as_of_time is None else float(as_of_time)
    frame = frame[frame["time"] <= snapshot]
    rows = []
    for user_id, group in frame.groupby("user_id", sort=False):
        row = {
            "user_id": user_id,
            "preferred_categories": _weighted_preferences(group, "_item_category", top_k),
            "preferred_subcategories": _weighted_preferences(
                group, "_item_subcategory", top_k),
        }
        prices = pd.to_numeric(group["_item_price"], errors="coerce")
        valid = prices.notna()
        row["event_price_mean"] = float(prices[valid].mean()) if valid.any() else 0.0
        row["event_price_std"] = float(prices[valid].std(ddof=0)) if valid.any() else 0.0
        for event_type in PRICE_EVENT_TYPES:
            selected = prices[group["type"].fillna("").astype(str) == event_type].dropna()
            row["event_%s_price_mean" % event_type] = float(selected.mean()) \
                if not selected.empty else 0.0
            row["event_%s_price_std" % event_type] = float(selected.std(ddof=0)) \
                if not selected.empty else 0.0
        for suffix, seconds in WINDOWS:
            recent = group["time"] >= snapshot - seconds
            for event_type in ("click", "buy"):
                selected = prices[recent & (group["type"] == event_type)].dropna()
                row["event_%s_price_mean_%s" % (event_type, suffix)] = \
                    float(selected.mean()) if not selected.empty else 0.0
        click_mean = row["event_click_price_mean"]
        buy_mean = row["event_buy_price_mean"]
        row["event_buy_to_click_price_ratio"] = \
            buy_mean / click_mean if click_mean > 0 else 0.0
        long_mean = row["event_click_price_mean_30d"]
        recent_mean = row["event_click_price_mean_1d"]
        row["event_recent_to_long_click_price_ratio"] = \
            recent_mean / long_mean if long_mean > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def enrich_user_commerce_features(users, events, items, as_of_time=None):
    if users is None:
        return None
    result = users.copy()
    columns = commerce_feature_columns()
    aggregated = aggregate_user_commerce_features(events, items, as_of_time)
    result = result.drop(columns=[name for name in columns if name in result], errors="ignore")
    if not aggregated.empty:
        result = result.merge(aggregated, how="left", left_on="id", right_on="user_id") \
            .drop(columns=["user_id"], errors="ignore")
    for name in columns:
        if name not in result:
            result[name] = "" if name.startswith("preferred_") else 0.0
        if name.startswith("preferred_"):
            result[name] = result[name].fillna("")
        else:
            result[name] = pd.to_numeric(result[name], errors="coerce").fillna(0.0)
    return result
