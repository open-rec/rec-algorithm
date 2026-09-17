"""Shared item-content materialization for offline training and online scoring."""

import numpy as np
import pandas as pd


def _epoch_seconds(values):
    """Parse epoch seconds/milliseconds or timestamp strings into UTC seconds."""
    numeric = pd.to_numeric(values, errors="coerce")
    seconds = numeric.where(numeric.abs() < 100_000_000_000, numeric / 1000)
    missing = seconds.isna()
    if missing.any():
        parsed = pd.to_datetime(values[missing], errors="coerce", utc=True)
        seconds.loc[missing] = parsed.astype("int64") / 1_000_000_000
        seconds.loc[missing & parsed.isna()] = np.nan
    return seconds


def enrich_item_content_features(items, as_of_time=None):
    """Return item rows with canonical content fields and point-in-time age.

    ``as_of_time`` accepts one epoch-second value or a row-aligned sequence.
    The same function is used by training and rank-engine, which prevents the
    freshness formula and raw camelCase/snake_case aliases from drifting.
    """
    if items is None:
        return None
    result = items.copy()
    aliases = {
        "pubTime": "pub_time",
        "subCategory": "subcategory",
    }
    for source, destination in aliases.items():
        if destination not in result and source in result:
            result[destination] = result[source]
    for name in ("title", "category", "subcategory", "tags"):
        if name not in result:
            result[name] = ""
        result[name] = result[name].fillna("")
    if "pub_time" not in result:
        result["pub_time"] = np.nan
    published = _epoch_seconds(result["pub_time"])
    if as_of_time is None:
        if "_as_of_time" in result:
            as_of = _epoch_seconds(result["_as_of_time"])
        else:
            as_of = pd.Series(
                pd.Timestamp.now(tz="UTC").timestamp(), index=result.index
            )
    elif np.isscalar(as_of_time):
        as_of = pd.Series(float(as_of_time), index=result.index)
    else:
        as_of = pd.Series(as_of_time, index=result.index, dtype=float)
    age = ((as_of - published) / 3600.0).clip(lower=0)
    result["content_age_hours"] = age.fillna(0.0)
    return result
