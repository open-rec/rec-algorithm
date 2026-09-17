"""Candidate-aware Transformer ranker for single-objective recommendation."""

import torch
import torch.nn as nn


class CandidateAwareTransformerModel(nn.Module):
    """Encode a user's item history and attend to it with the candidate.

    Dense semantic item vectors carry content meaning while ``global_features``
    keeps OpenRec's existing point-in-time user, item, scene, and freshness
    features in the scoring path.
    """

    def __init__(
        self,
        global_dim,
        semantic_dim,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        max_history=50,
        dropout=0.1,
    ):
        super().__init__()
        if min(global_dim, semantic_dim, model_dim, num_heads, num_layers, max_history) < 1:
            raise ValueError("Transformer dimensions and layer counts must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.global_dim = int(global_dim)
        self.semantic_dim = int(semantic_dim)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.max_history = int(max_history)
        self.dropout = float(dropout)

        self.article_projection = nn.Linear(semantic_dim, model_dim)
        self.position = nn.Parameter(torch.empty(max_history, model_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.candidate_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(global_dim), nn.Linear(global_dim, model_dim), nn.GELU()
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def forward(self, global_features, candidate, history, history_mask):
        if history.shape[1] != self.max_history:
            raise ValueError("history length does not match max_history")
        candidate_state = self.article_projection(candidate)
        history_state = self.article_projection(history) + self.position.unsqueeze(0)
        empty = history_mask.all(dim=1)
        safe_mask = history_mask.clone()
        safe_mask[empty, 0] = False
        history_state = history_state.clone()
        history_state[empty, 0] = 0
        encoded = self.history_encoder(history_state, src_key_padding_mask=safe_mask)
        interest, _ = self.candidate_attention(
            candidate_state.unsqueeze(1),
            encoded,
            encoded,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        interest = interest.squeeze(1)
        # An entirely empty history would otherwise yield NaNs from softmax.
        interest = torch.where(empty.unsqueeze(1), torch.zeros_like(interest), interest)
        global_state = self.global_projection(global_features)
        fused = torch.cat(
            [candidate_state, interest, candidate_state * interest, global_state], dim=1
        )
        return torch.sigmoid(self.scorer(fused))

    def architecture(self):
        return {
            "global_dim": self.global_dim,
            "semantic_dim": self.semantic_dim,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_history": self.max_history,
            "dropout": self.dropout,
        }
