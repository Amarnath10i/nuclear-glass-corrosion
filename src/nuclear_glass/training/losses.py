"""Physics-informed loss functions for glass corrosion models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def data_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Relative L2 data fitting loss."""
    return ((y_pred - y_true) ** 2).sum() / ((y_true ** 2).sum() + 1e-8)


def mass_conservation_residual(
    C: torch.Tensor,
    D: float = 1.0e-22,
    dx: float = 1.0e-6,
    dt: float = 86400.0,
) -> torch.Tensor:
    """PDE residual: ||dC/dt - D * d^2C/dx^2||^2. C shape: (batch, time, n_x)."""
    dC_dt   = torch.diff(C, dim=1) / dt
    d2C_dx2 = (C[:, :, 2:] - 2 * C[:, :, 1:-1] + C[:, :, :-2]) / dx**2
    residual = dC_dt[:, :, 1:-1] - D * d2C_dx2[:, :-1, :]
    return residual.pow(2).mean()


def tst_rate_residual(
    rate_pred: torch.Tensor,
    SI: torch.Tensor,
    temperature_K: torch.Tensor,
    k0: float = 1.0e-10,
    Ea_kJ_mol: float = 72.0,
) -> torch.Tensor:
    """Penalise violation of TST affinity-based rate law."""
    R = 8.314
    Ea = Ea_kJ_mol * 1e3
    k_T = k0 * torch.exp(-Ea / (R * temperature_K))
    A   = -2.303 * R * temperature_K * SI
    rate_tst = k_T * (1.0 - torch.exp(-A / (R * temperature_K)))
    return F.mse_loss(rate_pred, rate_tst.clamp(min=0))


def thermodynamic_monotonicity_loss(rate: torch.Tensor, SI: torch.Tensor) -> torch.Tensor:
    """Enforce that rate decreases as SI -> 0 (approach to equilibrium)."""
    grad = torch.diff(rate, dim=-1) / (torch.diff(SI, dim=-1) + 1e-8)
    violations = F.relu(grad)
    return violations.mean()


def combined_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    pde_residual: torch.Tensor,
    tst_residual: torch.Tensor,
    lambda_pde: float = 1.0,
    lambda_tst: float = 0.5,
) -> torch.Tensor:
    """Weighted sum of data + physics losses."""
    return data_loss(y_pred, y_true) + lambda_pde * pde_residual + lambda_tst * tst_residual

