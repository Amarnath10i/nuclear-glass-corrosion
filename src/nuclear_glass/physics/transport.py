"""
Reactive transport equations - diffusion-reaction in glass corrosion.

Implements 1-D finite-difference solver for:
    dC/dt = div(D*grad(C)) + R(C)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_diffusion_matrix(n: int, D: float, dx: float) -> NDArray[np.float64]:
    """Build tri-diagonal diffusion matrix for 1-D FD scheme."""
    alpha = D / dx**2
    diag = np.full(n, -2.0 * alpha)
    off  = np.full(n - 1, alpha)
    A = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    A[0, 0] = -alpha; A[0, 1] = alpha
    A[-1, -1] = 0.0;  A[-1, -2] = 0.0
    return A


def solve_transport_1d(
    C0: NDArray[np.float64],
    D: float,
    dx: float,
    dt: float,
    n_steps: int,
    reaction_fn=None,
) -> NDArray[np.float64]:
    """Explicit Euler 1-D reactive transport solver."""
    n = len(C0)
    A = build_diffusion_matrix(n, D, dx)
    C = C0.copy().astype(np.float64)
    history = np.empty((n_steps + 1, n), dtype=np.float64)
    history[0] = C

    for step in range(n_steps):
        R = reaction_fn(C) if reaction_fn is not None else np.zeros(n)
        C = C + dt * (A @ C + R)
        C = np.maximum(C, 0.0)
        history[step + 1] = C

    return history


def silicic_acid_saturation(
    C_Si_mol_m3: float | np.ndarray,
    temperature_K: float,
) -> np.ndarray:
    """Compute saturation index for amorphous silica. SI = log10(C_Si / Ksp)."""
    log_Ksp = -0.338 - 7.889e-4 * temperature_K
    Ksp_mol_m3 = (10.0 ** log_Ksp) * 1e3
    C = np.asarray(C_Si_mol_m3, dtype=float)
    return np.log10(np.maximum(C, 1e-30) / Ksp_mol_m3)
