"""Homogenisation / upscaling utilities for multi-scale glass corrosion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def volume_average_diffusivity(D_phases: NDArray, volume_fractions: NDArray) -> float:
    phi = np.asarray(volume_fractions)
    D   = np.asarray(D_phases)
    if not np.isclose(phi.sum(), 1.0, atol=0.01):
        raise ValueError(f"Volume fractions must sum to 1, got {phi.sum():.3f}")
    return float((phi * D).sum())


def hashin_shtrikman_bounds(D1: float, D2: float, phi2: float) -> tuple[float, float]:
    """Hashin-Shtrikman bounds for two-phase composite diffusivity."""
    phi1 = 1.0 - phi2
    D_lower = D1 + phi2 / (1.0 / (D2 - D1) + phi1 / (3.0 * D1))
    D_upper = D2 + phi1 / (1.0 / (D1 - D2) + phi2 / (3.0 * D2))
    if D_lower > D_upper:
        D_lower, D_upper = D_upper, D_lower
    return D_lower, D_upper


def kozeny_carman_permeability(porosity: float, grain_diameter_m: float, kozeny_const: float = 5.0) -> float:
    """Kozeny-Carman permeability [m^2]."""
    phi = porosity
    return grain_diameter_m**2 * phi**3 / (kozeny_const * (1.0 - phi) ** 2)


def representative_volume_element_size(heterogeneity_length_m: float, rve_factor: float = 20.0) -> float:
    return rve_factor * heterogeneity_length_m
