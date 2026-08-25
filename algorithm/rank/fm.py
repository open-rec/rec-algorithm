from pathlib import Path

import torch
import torch.nn as nn

from algorithm.feature.feature_space import FeatureSpace
from algorithm.rank.lr import LRRecModel
from algorithm.utils.file_util import DEFAULT_SCENE, feature_path, rank_model_path


MODEL_FILENAME = "fm.pth"
FEATURE_FILENAME = "fm.features.json"


class FMModel(nn.Module):
    """Second-order factorization machine over the shared, flattened FeatureSpace vector."""

    def __init__(self, dim=10, factor_dim=8):
        super().__init__()
        if factor_dim < 1:
            raise ValueError("factor_dim must be positive")
        self.dim = dim
        self.factor_dim = factor_dim
        self.linear = nn.Linear(in_features=dim, out_features=1)
        self.factors = nn.Parameter(torch.empty(dim, factor_dim))
        nn.init.xavier_uniform_(self.factors)

    def forward(self, x):
        linear = self.linear(x)
        # 1/2 * sum_f((sum_i v_if*x_i)^2 - sum_i(v_if*x_i)^2), calculated without
        # materialising every feature pair. This works for both sparse one-hot and numeric inputs.
        projected = torch.matmul(x, self.factors)
        squared_projected = projected.pow(2)
        projected_squared = torch.matmul(x.pow(2), self.factors.pow(2))
        interactions = .5 * (squared_projected - projected_squared).sum(dim=1, keepdim=True)
        return torch.sigmoid(linear + interactions)


class FMRecModel(LRRecModel):
    """FM training and scoring with the same dataset and online feature contract as LR."""

    def __init__(self, user_feature=None, item_feature=None, events=None, feature_space=None,
                 scene=DEFAULT_SCENE, model_file=None, feature_file=None, factor_dim=8):
        model_file = model_file or rank_model_path(scene) / MODEL_FILENAME
        feature_file = feature_file or feature_path(scene) / FEATURE_FILENAME
        super().__init__(user_feature=user_feature, item_feature=item_feature, events=events,
                         feature_space=feature_space, scene=scene, model_file=model_file,
                         feature_file=feature_file, model_type="fm")
        self.model = FMModel(dim=self.dataset.feature_dim, factor_dim=factor_dim)

    def load(self):
        feature_file = Path(self.feature_file)
        if feature_file.exists():
            self.dataset.rebind_space(FeatureSpace.load(feature_file))

        state = torch.load(self.model_file, map_location="cpu")
        checkpoint_dim = state.get("linear.weight", torch.empty(1, 0)).shape[-1]
        factors = state.get("factors")
        if factors is None or factors.ndim != 2:
            raise ValueError("FM checkpoint does not contain a two-dimensional factors tensor")
        if checkpoint_dim != self.dataset.feature_dim or factors.shape[0] != checkpoint_dim:
            raise ValueError(
                f"{self.model_file} was trained with dim={checkpoint_dim}, but the current feature "
                f"space yields dim={self.dataset.feature_dim}. Use its {FEATURE_FILENAME} sidecar.")
        self.model = FMModel(dim=checkpoint_dim, factor_dim=factors.shape[1])
        self.model.load_state_dict(state)
        self.model.eval()
