import math

import pandas as pd

from algorithm.recall.content_i2i import ContentBasedI2I


def items():
    return pd.DataFrame([
        ("a", "movie/action", "hero,space", "space hero"),
        ("b", "movie/action", "hero", "another hero"),
        ("c", "book/history", "ancient", "old world"),
    ], columns=["id", "category", "tags", "title"])


def test_content_recall_uses_tfidf_cosine_and_omits_unrelated_items():
    model = ContentBasedI2I(items=items(), recall_size=10, cut_size=10)
    neighbours = dict(model.dump_i2i(10)["a"])

    assert set(neighbours) == {"b"}
    assert 0 < neighbours["b"] < 1
    assert model.recall(item_triggers=["a"])[0].item == "b"


def test_content_similarity_is_symmetric():
    neighbours = ContentBasedI2I(items=items()).dump_i2i(10)

    assert math.isclose(dict(neighbours["a"])["b"], dict(neighbours["b"])["a"])
