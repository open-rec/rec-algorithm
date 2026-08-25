"""Generate deterministic, learnable OpenRec user/item/event fixtures.

The event population models an impression funnel rather than drawing event types independently:
every trace starts with ``expose`` and may progress through ``stay``, ``click``, ``collect`` and
``buy``. User interests, item categories/tags, popularity and position affect conversion, giving
recall and rank jobs a signal that is stronger than random noise.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path


USER_HEADERS = [
    "id", "device_id", "name", "gender", "age", "country", "city", "phone", "tags",
    "register_time", "login_time", "ext_fields",
]
ITEM_HEADERS = [
    "id", "title", "category", "tags", "scene", "pub_time", "modify_time", "expire_time",
    "status", "weight", "ext_fields",
]
EVENT_HEADERS = [
    "id", "user_id", "item_id", "trace_id", "scene", "type", "value", "time", "is_login",
    "ext_fields",
]


def _write(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def generate(output, user_count=10000, item_count=10000, impression_count=100000,
             scene_count=3, days=30, seed=42, end_time=1704067200):
    if min(user_count, item_count, impression_count, scene_count, days) < 1:
        raise ValueError("counts and days must be positive")
    rng = random.Random(seed)
    output = Path(output)
    categories = ["category_%d" % i for i in range(max(8, min(64, item_count // 20 + 1)))]
    tags = ["tag_%d" % i for i in range(max(16, min(128, item_count // 10 + 1)))]
    countries = ("CN", "US", "JP", "SG")

    users = []
    user_profiles = []
    for i in range(user_count):
        interests = rng.sample(categories, min(3, len(categories)))
        interest_tags = rng.sample(tags, min(4, len(tags)))
        activity = rng.betavariate(2.0, 5.0)
        register_time = end_time - rng.randint(days * 86400, 720 * 86400)
        login_time = end_time - rng.randint(0, 7 * 86400)
        users.append({
            "id": "user_%d" % i, "device_id": "device_%d" % i, "name": "user_%d" % i,
            "gender": i % 2, "age": rng.randint(16, 70), "country": rng.choice(countries),
            "city": "city_%d" % rng.randrange(20), "phone": "phone_%d" % i,
            # Categories are included deliberately: FM can learn user-interest × item-category
            # interactions from fields that are part of the deployed feature contract.
            "tags": ",".join(interests + interest_tags), "register_time": register_time,
            "login_time": login_time,
            "ext_fields": json.dumps({"interest_categories": interests,
                                      "activity": round(activity, 4)}, separators=(",", ":")),
        })
        user_profiles.append((set(interests), set(interest_tags), activity))

    items = []
    item_profiles = []
    scene_items = [[] for _ in range(scene_count)]
    for i in range(item_count):
        scene_index = i % scene_count
        category = rng.choice(categories)
        item_tags = set(rng.sample(tags, min(3, len(tags))))
        popularity = rng.betavariate(1.2, 4.0)
        generated_pub_time = end_time - rng.randint(0, days * 86400)
        generated_modify_time = generated_pub_time + rng.randint(
            0, max(1, end_time - generated_pub_time))
        # Fifty deterministic fresh items per scene keep NewNode useful after user-level expose
        # filtering. Draw the ordinary timestamps first so this fixture does not perturb the rest
        # of the seeded random population.
        fresh_fixture = i < scene_count * 50
        pub_time = end_time if fresh_fixture else generated_pub_time
        modify_time = end_time if fresh_fixture else generated_modify_time
        weight = max(1, min(10, int(round(1 + popularity * 9))))
        items.append({
            "id": "item_%d" % i, "title": "%s item %d" % (category, i),
            "category": category, "tags": ",".join(sorted(item_tags)),
            "scene": "scene_%d" % scene_index, "pub_time": pub_time,
            "modify_time": modify_time,
            "expire_time": end_time + rng.randint(30, 365) * 86400, "status": 1,
            "weight": weight,
            "ext_fields": json.dumps({"quality": round(popularity, 4),
                                      "fresh_fixture": fresh_fixture}, separators=(",", ":")),
        })
        item_profiles.append((category, item_tags, popularity, scene_index))
        scene_items[scene_index].append(i)

    events = []
    event_id = 0
    start_time = end_time - days * 86400
    scripted_per_scene = [min(24, len(candidates)) for candidates in scene_items]
    scripted_count = sum(scripted_per_scene)
    if scripted_count >= impression_count:
        raise ValueError("impression count is too small for cross-scene user_0 fixtures")
    random_impressions = impression_count - scripted_count
    for impression in range(random_impressions):
        user_index = rng.randrange(user_count)
        interests, interest_tags, activity = user_profiles[user_index]
        scene_index = rng.randrange(scene_count)
        candidates = scene_items[scene_index]
        item_index = rng.choice(candidates)
        category, item_tags, popularity, _ = item_profiles[item_index]
        category_match = category in interests
        tag_overlap = len(item_tags & interest_tags)
        position = rng.randint(1, 30)
        position_effect = 0.75 + 0.25 * math.exp(-(position - 1) / 18.0)
        weight = items[item_index]["weight"]
        # The dominant terms are deliberately present in the deployed Feature Set. LR learns the
        # monotonic quality/weight prior; FM additionally learns interest-category/tag crosses.
        logit = (-5.2 + 0.85 * weight + (2.2 if category_match else 0.0)
                 + 0.55 * tag_overlap + 0.25 * activity)
        click_probability = (1.0 / (1.0 + math.exp(-logit))) * position_effect
        relevance = min(1.0, click_probability * 1.8 + 0.10)
        timestamp = start_time + int((impression + rng.random()) * days * 86400 /
                                     random_impressions)
        trace_id = "trace_%d" % impression

        def append(kind, value, offset):
            nonlocal event_id
            events.append({
                "id": "event_%d" % event_id, "user_id": "user_%d" % user_index,
                "item_id": "item_%d" % item_index, "trace_id": trace_id,
                "scene": "scene_%d" % scene_index, "type": kind, "value": value,
                "time": timestamp + offset, "is_login": 1,
                "ext_fields": json.dumps({"position": position}, separators=(",", ":")),
            })
            event_id += 1

        append("expose", 1, 0)
        watch_ratio = min(2.0, max(0.0, rng.gauss(0.12 + relevance, 0.22)))
        if rng.random() < min(0.90, 0.08 + relevance * 0.72 * position_effect):
            append("stay", round(watch_ratio * 60, 3), 1)
        clicked = rng.random() < click_probability
        if clicked:
            append("click", 1, 2)
            if rng.random() < 0.015 + relevance * 0.18:
                append("collect", 1, 3)
            if rng.random() < 0.004 + relevance * popularity * 0.07:
                append("buy", 1, 4)

    # Stable multi-scene fixtures for demos and smoke tests. user_0 receives a sequence of high
    # quality clicks plus low-quality unclicked exposures in every scene, guaranteeing useful I2I
    # triggers and non-empty per-scene ranking history without special-casing downstream code.
    fixture_time = start_time + int(days * 86400 * .55)
    fixture_index = random_impressions
    user0_interests, _, _ = user_profiles[0]
    for scene_index, fixture_size in enumerate(scripted_per_scene):
        candidates = list(scene_items[scene_index])
        candidates.sort(key=lambda index: (
            item_profiles[index][0] in user0_interests, items[index]["weight"], -index), reverse=True)
        positive_size = max(4, fixture_size // 2)
        selected = candidates[:positive_size] + list(reversed(candidates[-(fixture_size - positive_size):]))
        for position, item_index in enumerate(selected, 1):
            clicked = position <= positive_size
            timestamp = fixture_time + scene_index * 1000 + position * 10
            trace_id = "trace_%d" % fixture_index
            fixture_index += 1

            def append_fixture(kind, value, offset):
                nonlocal event_id
                events.append({
                    "id": "event_%d" % event_id, "user_id": "user_0",
                    "item_id": "item_%d" % item_index, "trace_id": trace_id,
                    "scene": "scene_%d" % scene_index, "type": kind, "value": value,
                    "time": timestamp + offset, "is_login": 1,
                    "ext_fields": json.dumps({"position": position, "fixture": True},
                                             separators=(",", ":")),
                })
                event_id += 1

            append_fixture("expose", 1, 0)
            if clicked or position % 3 == 0:
                append_fixture("stay", 55 if clicked else 8, 1)
            if clicked:
                append_fixture("click", 1, 2)
                if position % 3 == 0:
                    append_fixture("collect", 1, 3)
                if position <= 2:
                    append_fixture("buy", 1, 4)

    _write(output / "user.csv", USER_HEADERS, users)
    _write(output / "item.csv", ITEM_HEADERS, items)
    _write(output / "event.csv", EVENT_HEADERS, events)
    return {"users": len(users), "items": len(items), "impressions": impression_count,
            "events": len(events), "seed": seed, "end_time": end_time}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="../example/data/test")
    parser.add_argument("--users", type=int, default=10000)
    parser.add_argument("--items", type=int, default=10000)
    parser.add_argument("--impressions", type=int, default=100000)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end-time", type=int, default=1704067200,
                        help="fixed UTC Unix timestamp; keep it stable for reproducible fixtures")
    args = parser.parse_args()
    result = generate(args.output, args.users, args.items, args.impressions, args.scenes,
                      args.days, args.seed, args.end_time)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
