import pandas as pd

from algorithm.feature.commerce_feature import aggregate_user_commerce_features


def test_category_and_price_features_use_item_version_at_event_time():
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "i", "type": "expose", "value": 1, "time": 100},
        {"user_id": "u", "item_id": "i", "type": "click", "value": 1, "time": 200},
        {"user_id": "u", "item_id": "j", "type": "collect", "value": 1, "time": 300},
        {"user_id": "u", "item_id": "j", "type": "buy", "value": 1, "time": 400},
    ])
    items = pd.DataFrame([
        {"id": "i", "category": "old", "subcategory": "a",
         "ext_fields": '{"price": 10}', "_effective_time": 0},
        {"id": "i", "category": "future", "subcategory": "b",
         "ext_fields": '{"price": 100}', "_effective_time": 250},
        {"id": "j", "category": "buy-category", "subcategory": "c",
         "ext_fields": {"unitPrice": 30}, "_effective_time": 0},
    ])
    row = aggregate_user_commerce_features(events, items, 400).iloc[0]
    assert row.preferred_categories == "buy-category,old"
    assert row.preferred_subcategories == "c,a"
    assert row.event_expose_price_mean == 10
    assert row.event_click_price_mean == 10
    assert row.event_collect_price_mean == 30
    assert row.event_buy_price_mean == 30
    assert row.event_buy_to_click_price_ratio == 3


def test_price_windows_and_deleted_item_versions_do_not_leak():
    events = pd.DataFrame([
        {"user_id": "u", "item_id": "old", "type": "click", "time": 1},
        {"user_id": "u", "item_id": "recent", "type": "click", "time": 200000},
        {"user_id": "u", "item_id": "deleted", "type": "buy", "time": 200000},
    ])
    items = pd.DataFrame([
        {"id": "old", "price": 10, "_effective_time": 0},
        {"id": "recent", "price": 20, "_effective_time": 0},
        {"id": "deleted", "price": 999, "_effective_time": 0, "_operation": "INSERT"},
        {"id": "deleted", "price": 999, "_effective_time": 100, "_operation": "DELETE"},
    ])
    row = aggregate_user_commerce_features(events, items, 200000).iloc[0]
    assert row.event_click_price_mean == 15
    assert row.event_click_price_mean_1d == 20
    assert row.event_click_price_mean_30d == 15
    assert row.event_recent_to_long_click_price_ratio == 20 / 15
    assert row.event_buy_price_mean == 0
