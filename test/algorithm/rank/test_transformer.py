import pytest
import torch

from algorithm.rank.transformer import CandidateAwareTransformerModel


def test_candidate_aware_transformer_shape_and_empty_history():
    model = CandidateAwareTransformerModel(
        global_dim=6, semantic_dim=8, model_dim=16, num_heads=4,
        num_layers=1, max_history=3, dropout=0,
    )
    global_features = torch.randn(4, 6)
    candidate = torch.randn(4, 8)
    history = torch.randn(4, 3, 8)
    mask = torch.tensor([
        [False, False, True], [False, True, True],
        [True, True, True], [False, False, False],
    ])
    scores = model(global_features, candidate, history, mask)
    assert scores.shape == (4, 1)
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_candidate_aware_transformer_validates_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        CandidateAwareTransformerModel(2, 3, model_dim=10, num_heads=4)
