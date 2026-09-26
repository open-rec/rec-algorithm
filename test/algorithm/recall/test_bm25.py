from algorithm.recall.bm25 import BM25Recall, tokenize


def test_bm25_recall_supports_field_weights_and_exclusion():
    recall = BM25Recall({
        "exact": {"title": "blue train", "tags": "jazz"},
        "tagged": {"title": "night", "tags": "blue jazz"},
        "other": {"title": "rock", "tags": "guitar"},
    }, field_weights={"title": 3.0, "tags": 1.0})

    assert [value.item for value in recall.recall("blue", exclude=["exact"])] == ["tagged"]
    assert recall.recall("") == []


def test_tokenize_accepts_text_and_terms():
    assert tokenize("Jazz, BLUE!") == ["jazz", "blue"]
    assert tokenize(["Jazz", "Blue Train"]) == ["jazz", "blue train"]
