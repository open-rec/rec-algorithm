"""Materialize user/item snapshot feature tables for the online feature store."""

import argparse
from pathlib import Path

import pandas as pd

from algorithm.feature.event_feature import enrich_entity_features


def materialize(user_file, item_file, event_file, output_dir, as_of_time=None):
    users = pd.read_csv(user_file)
    items = pd.read_csv(item_file)
    events = pd.read_csv(event_file)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    user_features = enrich_entity_features(users, events, "user", as_of_time)
    item_features = enrich_entity_features(items, events, "item", as_of_time)
    user_features.to_csv(str(output / "user_feature.csv"), index=False)
    item_features.to_csv(str(output / "item_feature.csv"), index=False)
    return user_features, item_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--item", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of-time", type=int, default=None)
    args = parser.parse_args()
    users, items = materialize(args.user, args.item, args.event, args.output, args.as_of_time)
    print("materialized {} users and {} items into {}".format(len(users), len(items), args.output))


if __name__ == "__main__":
    main()
