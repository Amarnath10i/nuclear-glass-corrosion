"""
DeepONet architecture for operator learning.

Reference:
    Lu, L. et al. (2021). Nature Machine Intelligence 3, 218-229.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, activation: type[nn.Module] = nn.GELU) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), activation()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Branch-Trunk DeepONet for solution operator learning."""

    def __init__(
        self,
        branch_input_dim: int = 100,
        trunk_input_dim: int = 2,
        p: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 5,
    ) -> None:
        super().__init__()
        self.branch = _mlp(branch_input_dim, hidden_dim, p, n_layers)
        self.trunk  = _mlp(trunk_input_dim,  hidden_dim, p, n_layers)
        self.bias   = nn.Parameter(torch.zeros(1))

    def forward(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b = self.branch(v)  # (batch, p)
        t = self.trunk(y)   # (n_eval, p)
        return torch.einsum("bp,ep->be", b, t) + self.bias

    @staticmethod
    def data_driven_loss(u_pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = ((u_pred - u_true).norm(dim=-1) / (u_true.norm(dim=-1) + 1e-8)).mean()
        return loss
