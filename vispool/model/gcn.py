from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedGCNConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.linear.weight)

    def forward(self, vvgs: torch.Tensor, token_embs: torch.Tensor) -> Any:
        out = vvgs @ token_embs
        out = self.layer_norm(out)
        out = self.linear(out)
        out = F.relu(out)
        return out


class OverallGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.gcn1 = BatchedGCNConv(in_dim, hidden_dim)
        self.gcn2 = BatchedGCNConv(hidden_dim, out_dim)
        self.pool = pool

    def forward(self, vvgs: torch.Tensor, token_embs: torch.Tensor) -> Any:
        out = self.gcn1(vvgs, token_embs)
        out = self.dropout(out)
        out = self.gcn2(vvgs, out)
        out = self.dropout(out)

        if self.pool == "mean":
            out = out.mean(dim=-2)
        elif self.pool == "max":
            out, _ = out.max(dim=-2)
        else:
            out = out[..., 0, :]  # cls token

        if self.out_dim > 1:
            out = F.log_softmax(out, dim=-1)
        return out
