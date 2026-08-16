"""
Regression tests for the i2i recall contract.

gen_seq() is the only part that needs pandas, so stubbing it lets the similarity computation and the
merge logic be tested from any directory — unlike the CSV-driven tests next door, which only pass
when run from their own directory.
"""
from algorithm.recall.i2i import ItemBasedI2I

SEQUENCES = {
    "u1": [("a", 1), ("b", 2), ("c", 3), ("d", 4)],
    "u2": [("a", 1), ("b", 2), ("e", 5)],
    "u3": [("a", 1), ("c", 3), ("d", 4), ("f", 6)],
    "u4": [("b", 2), ("c", 3), ("g", 7)],
}


class StubI2I(ItemBasedI2I):
    """Feeds fixed sequences in place of the pandas-backed gen_seq()."""

    def __init__(self, sequences=None, **kwargs):
        super().__init__(events=None, **kwargs)
        self._sequences = sequences or SEQUENCES

    def gen_seq(self):
        return self._sequences


def test_respects_recall_size():
    recall_size = 3
    items = StubI2I(recall_size=recall_size, cut_size=10).recall(item_triggers=["a", "b"])
    assert len(items) <= recall_size


def test_merges_duplicates_across_triggers():
    items = StubI2I(recall_size=50, cut_size=10).recall(item_triggers=["a", "b"])
    ids = [i.item for i in items]
    assert len(ids) == len(set(ids))


def test_never_recalls_a_trigger():
    triggers = ["a", "b"]
    items = StubI2I(recall_size=50, cut_size=10).recall(item_triggers=triggers)
    assert not set(triggers) & {i.item for i in items}


def test_sorted_by_score_desc():
    scores = [i.score for i in StubI2I(recall_size=50, cut_size=10).recall(item_triggers=["a", "b"])]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_keeps_best_score():
    engine = StubI2I(recall_size=50, cut_size=10)
    merged = {i.item: i.score for i in engine.recall(item_triggers=["a", "b"])}
    neighbours = engine.dump_i2i(cut_size=10)

    for item, score in merged.items():
        best = max(s for t in ("a", "b") for i, s in neighbours.get(t, []) if i == item)
        assert score == best


def test_similarity_is_cached():
    engine = StubI2I(recall_size=5, cut_size=10)
    calls = []
    stubbed = engine.gen_seq

    def counting():
        calls.append(1)
        return stubbed()

    engine.gen_seq = counting
    engine.recall(item_triggers=["a"])
    engine.recall(item_triggers=["b"])
    engine.dump_i2i(cut_size=5)

    assert len(calls) == 1


def test_unknown_trigger_yields_nothing():
    assert StubI2I(recall_size=10, cut_size=10).recall(item_triggers=["nonexistent"]) == []
