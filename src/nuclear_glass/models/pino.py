"""
Physics-Informed Neural Operator (PINO) for glass corrosion.

Architecture: Fourier Neural Operator (FNO) backbone with physics-residual losses.

Reference:
    Li, Z. et al. (2021). Physics-Informed Neural Operator. arXiv:2111.03794.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1-D Fourier layer for operator learning."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.out_channels, N // 2 + 1, dtype=torch.cfloat, device=x.device)
        w = torch.view_as_complex(self.weights)
        out_ft[:, :, : self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, : self.modes], w)
        return torch.fft.irfft(out_ft, n=N)


class FNOBlock1d(nn.Module):
    """Single FNO residual block."""

    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.spectral  = SpectralConv1d(channels, channels, modes)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm      = nn.InstanceNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class PINO(nn.Module):
    """Physics-Informed Neural Operator for corrosion rate prediction."""

    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 4,
        modes: int = 16,
    ) -> None:
        super().__init__()
        self.lift    = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.blocks  = nn.Sequential(*[FNOBlock1d(hidden_dim, modes) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, out_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        x = self.blocks(x)
        return self.project(x)

    def physics_residual(self, x: torch.Tensor, y_pred: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Mass-conservation PDE residual penalty."""
        dy_dt = torch.diff(y_pred, dim=-1) / dt
        d2y_dx2 = y_pred[:, :, 2:] - 2 * y_pred[:, :, 1:-1] + y_pred[:, :, :-2]
        D_eff = 1.0e-22
        residual = dy_dt[:, :, :-1] - D_eff * d2y_dx2
        return residual.pow(2).mean()
