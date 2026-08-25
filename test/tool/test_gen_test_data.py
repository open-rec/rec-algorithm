import csv
from collections import Counter

from tool.gen_test_data import generate


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_generate_is_deterministic_and_models_an_impression_funnel(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    stats = generate(first, user_count=40, item_count=80, impression_count=2000,
                     scene_count=3, seed=7)
    generate(second, user_count=40, item_count=80, impression_count=2000,
             scene_count=3, seed=7)

    assert (first / "event.csv").read_bytes() == (second / "event.csv").read_bytes()
    users, items, events = read(first / "user.csv"), read(first / "item.csv"), read(first / "event.csv")
    assert len(users) == 40 and len(items) == 80 and stats["impressions"] == 2000
    counts = Counter(row["type"] for row in events)
    assert counts["expose"] == 2000
    assert 0 < counts["buy"] < counts["collect"] < counts["click"] < counts["expose"]
    item_scene = {row["id"]: row["scene"] for row in items}
    assert all(item_scene[row["item_id"]] == row["scene"] for row in events)
    assert all(row["tags"] and row["ext_fields"] != "{}" for row in users + items)
    newest = max(int(row["pub_time"]) for row in items)
    for scene in ("scene_0", "scene_1", "scene_2"):
        assert sum(row["scene"] == scene and int(row["pub_time"]) == newest
                   for row in items) >= 20
    for scene in ("scene_0", "scene_1", "scene_2"):
        user0 = [row for row in events if row["user_id"] == "user_0" and row["scene"] == scene]
        fixture_types = Counter(row["type"] for row in user0)
        assert fixture_types["expose"] >= 24 and fixture_types["click"] >= 12
        assert fixture_types["collect"] >= 4 and fixture_types["buy"] >= 2


def test_matching_interests_have_a_higher_click_rate(tmp_path):
    generate(tmp_path, user_count=100, item_count=200, impression_count=12000, seed=19)
    users = {row["id"]: row for row in read(tmp_path / "user.csv")}
    items = {row["id"]: row for row in read(tmp_path / "item.csv")}
    traces = {}
    for event in read(tmp_path / "event.csv"):
        traces.setdefault(event["trace_id"], []).append(event)
    buckets = {True: [0, 0], False: [0, 0]}
    for events in traces.values():
        exposed = events[0]
        interests = set(__import__("json").loads(users[exposed["user_id"]]["ext_fields"])
                        ["interest_categories"])
        matched = items[exposed["item_id"]]["category"] in interests
        buckets[matched][1] += 1
        buckets[matched][0] += any(event["type"] == "click" for event in events)
    assert buckets[True][0] / buckets[True][1] > buckets[False][0] / buckets[False][1]
