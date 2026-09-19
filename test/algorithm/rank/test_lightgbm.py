import numpy as np
import pytest

from algorithm.rank.lightgbm import LightGBMBinaryModel, LightGBMRankModel


def test_group_sizes_rejects_noncontiguous_queries():
    assert LightGBMRankModel.group_sizes(["a", "a", "b"]) == [2, 1]
    with pytest.raises(ValueError, match="contiguous"):
        LightGBMRankModel.group_sizes(["a", "b", "a"])


def test_ranker_round_trip(tmp_path):
    pytest.importorskip("lightgbm")
    x = np.array([[1, 0], [0, 1], [2, 0], [0, 2]], dtype=np.float32)
    y = np.array([1, 0, 1, 0])
    groups = np.array(["a", "a", "b", "b"])
    model = LightGBMRankModel(n_estimators=5, min_child_samples=1).fit(x, y, groups)
    path = tmp_path / "model.txt"
    model.save(path)
    score = model.predict_proba(x)
    restored = LightGBMRankModel.load(path).predict_proba(x)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
    assert np.allclose(score, restored)


def test_binary_round_trip(tmp_path):
    pytest.importorskip("lightgbm")
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])
    model = LightGBMBinaryModel(n_estimators=5, min_child_samples=1).fit(
        x, y, validation=(x, y, None)
    )
    path = tmp_path / "binary-model.txt"
    model.save(path)
    score = model.predict_proba(x)
    restored = LightGBMBinaryModel.load(path).predict_proba(x)
    assert np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
    assert np.allclose(score, restored)
