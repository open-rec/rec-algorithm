"""Materialize user/item snapshot feature tables for the online feature store."""

import argparse
from pathlib import Path

import pandas as pd

from algorithm.feature.event_feature import enrich_entity_features
from algorithm.feature.feature_space import FeatureSpace


def materialize(user_file, item_file, event_file, output_dir, as_of_time=None,
                feature_space_file=None):
    users = pd.read_csv(user_file)
    items = pd.read_csv(item_file)
    events = pd.read_csv(event_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    user_features = enrich_entity_features(users, events, "user", as_of_time)
    item_features = enrich_entity_features(items, events, "item", as_of_time)
    cutoff = int(as_of_time if as_of_time is not None else
                 pd.to_numeric(events.get("time"), errors="coerce").max())
    user_features.insert(1, "as_of_time", cutoff)
    item_features.insert(1, "as_of_time", cutoff)
    user_features.to_csv(str(output / "user_feature.csv"), index=False)
    item_features.to_csv(str(output / "item_feature.csv"), index=False)
    if feature_space_file:
        space = FeatureSpace().fit(user_features, item_features)
        feature_path = Path(feature_space_file)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        space.save(feature_path)
    return user_features, item_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of-time", type=int, default=None)
    parser.add_argument("--feature-space", default=None,
                        help="optional output path for the fitted FeatureSpace JSON")
    args = parser.parse_args()
    users, items = materialize(args.user, args.item, args.event, args.output, args.as_of_time,
                               args.feature_space)
    print("materialized {} users and {} items into {}".format(len(users), len(items), args.output))


if __name__ == "__main__":
    main()
