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
            publish_recall(frame, "i2i", "2026-08-20")
            raise AssertionError("expected empty release to fail")
        except ValueError as error:
            assert "empty i2i" in str(error)
    request.assert_not_called()
