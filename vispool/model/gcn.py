from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoolStrategy(Enum):
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"


class BatchedGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, layer_norm: bool = True) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_norm = nn.LayerNorm(out_dim) if layer_norm else None
        self.linear = nn.Linear(in_dim, out_dim)

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, vvgs: torch.Tensor, token_embs: torch.Tensor) -> Any:
        out = vvgs @ token_embs
        out = self.linear(out)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        out = F.relu(out)
        return out


class OverallGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pool: PoolStrategy = PoolStrategy.CLS,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.gcn1 = BatchedGCNConv(in_dim, hidden_dim, layer_norm)
        self.gcn2 = BatchedGCNConv(hidden_dim, hidden_dim // 2, layer_norm)
        self.linear = nn.Linear(hidden_dim // 2, out_dim)
        self.pool = pool

    def forward(self, vvgs: torch.Tensor, token_embs: torch.Tensor) -> Any:
        out = self.gcn1(vvgs, token_embs)
        out = self.dropout(out)
        out = self.gcn2(vvgs, out)
        out = self.dropout(out)
        out = self.linear(out)

        if self.pool == "mean":
            out = out.mean(dim=-2)
        elif self.pool == "max":
            out, _ = out.max(dim=-2)
        else:
            out = out[..., 0, :]  # cls token

        if self.out_dim > 1:
            out = F.log_softmax(out, dim=-1)
        return out
