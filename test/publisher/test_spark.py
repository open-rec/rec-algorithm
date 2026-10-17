from unittest.mock import Mock, call, patch

from publisher.spark import publish_recall


def test_publisher_delegates_index_lifecycle_to_console():
    frame = Mock()
    frame.count.return_value = 42
    responses = [
        {"index": "openrec-recall-hot-20260820-r001", "writable": False,
         "idempotent": True},
        {"index": "openrec-recall-hot-20260820-r001", "documents": 42},
    ]
    with patch("publisher.spark._console_request", side_effect=responses) as request:
        result = publish_recall(frame, "hot", "2026-08-20",
                                console_url="http://rec-console:8095")

    assert result["documents"] == 42
    assert request.call_args_list == [
        call("http://rec-console:8095", "/api/recall/releases/prepare", {
            "algorithm": "hot", "business_date": "2026-08-20", "revision": "r001",
        }),
        call("http://rec-console:8095", "/api/recall/releases/activate", {
            "algorithm": "hot", "index": "openrec-recall-hot-20260820-r001",
            "expected_documents": 42, "max_index_versions": 2,
        }),
    ]
    frame.repartition.assert_not_called()


def test_publisher_refuses_empty_release_before_calling_console():
    frame = Mock()
    frame.count.return_value = 0
    with patch("publisher.spark._console_request") as request:
        try:
            publish_recall(frame, "item-cf-i2i", "2026-08-20")
            raise AssertionError("expected empty release to fail")
        except ValueError as error:
            assert "empty item-cf-i2i" in str(error)
    request.assert_not_called()


def test_search_publishers_preserve_entity_and_text_without_score():
    import sys
    from types import SimpleNamespace
    for entity in ("item", "user"):
        kind = entity + "-search"
        frame = Mock()
        frame.count.return_value = 1
        row = SimpleNamespace(scene="s", text="hello", **{entity: "id-1"})
        frame.repartition.return_value.foreachPartition.side_effect = lambda write: write([row])
        client = Mock()
        actions = []
        client.helpers.bulk.side_effect = lambda es, rows: actions.extend(rows)
        with patch.dict(sys.modules, {"elasticsearch": client}), patch(
                "publisher.spark._console_request", side_effect=[{"index": kind + "-staging"}, {}]):
            publish_recall(frame, kind, "2026-10-17")
        assert actions == [{"_index": kind + "-staging", "_id": "s:id-1",
                            "_source": {"scene": "s", entity: "id-1", "text": "hello"}}]
