import pandas as pd
import torch

from algorithm.feature.item_feature import ItemFeature
from algorithm.feature.user_feature import UserFeature
from algorithm.rank.fm import FMModel, FMRecModel


def frames():
    users = pd.DataFrame([
        {"id": "u1", "country": "CN", "city": "HZ", "gender": 1, "age": 20, "tags": "tech"},
        {"id": "u2", "country": "CN", "city": "SH", "gender": 0, "age": 30, "tags": "book"},
    ])
    items = pd.DataFrame([
        {"id": "i1", "category": "tech", "scene": "home", "weight": 1},
        {"id": "i2", "category": "book", "scene": "home", "weight": 2},
    ])
    events = pd.DataFrame([
        {"user_id": "u1", "item_id": "i1", "type": "click"},
        {"user_id": "u1", "item_id": "i2", "type": "expose"},
        {"user_id": "u2", "item_id": "i1", "type": "expose"},
        {"user_id": "u2", "item_id": "i2", "type": "click"},
    ])
    return users, items, events


def test_fm_forward_matches_explicit_pairwise_interactions():
    model = FMModel(dim=3, factor_dim=2)
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[.1, .2, .3]]))
        model.linear.bias.copy_(torch.tensor([.4]))
        model.factors.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
    x = torch.tensor([[1., 2., 3.]])
    pairwise = sum(
        torch.dot(model.factors[i], model.factors[j]) * x[0, i] * x[0, j]
        for i in range(3) for j in range(i + 1, 3))
    expected = torch.sigmoid(model.linear(x).reshape(()) + pairwise)
    assert torch.allclose(model(x).reshape(()), expected)


def test_fm_reuses_feature_sidecar_and_round_trips(tmp_path):
    users, items, events = frames()
    model_file, feature_file = tmp_path / "fm.pth", tmp_path / "fm.features.json"
    model = FMRecModel(UserFeature(users, events), ItemFeature(items, events), events,
                       model_file=model_file, feature_file=feature_file, factor_dim=4)
    model.train(epoch_num=1, batch_size=2, val_ratio=0)
    model.save()
    expected = model.score("u1", ["i1", "i2"])

    loaded = FMRecModel(UserFeature(users, events), ItemFeature(items, events), events,
                        model_file=model_file, feature_file=feature_file, factor_dim=1)
    loaded.load()
    assert loaded.model.factor_dim == 4
    assert loaded.score("u1", ["i1", "i2"]) == expected
