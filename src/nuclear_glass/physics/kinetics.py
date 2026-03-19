"""
Dissolution kinetics module.

Implements:
  - Stage I (initial dissolution) - far from saturation
  - Stage II (residual rate) - passivated gel layer regime
  - Stage III (resumption) - alteration layer transformation
"""

from __future__ import annotations

import numpy as np


def stage1_rate(k0: float, temperature_K: float, Ea_kJ_mol: float, pH: float) -> float:
    """Forward dissolution rate in Stage I (far from saturation) [m/s]."""
    from nuclear_glass.physics.thermodynamics import arrhenius_rate_constant
    k = arrhenius_rate_constant(k0, Ea_kJ_mol, temperature_K)
    return k * (10.0 ** (-pH)) ** (-0.4)


def stage2_residual_rate(
    r1: float,
    gel_thickness_m: float,
    diffusivity_m2_s: float = 1.0e-22,
) -> float:
    """Residual rate limited by diffusion through gel layer [m/s]."""
    if gel_thickness_m <= 0:
        return r1
    return diffusivity_m2_s / gel_thickness_m


def stage3_resumption_rate(r2: float, amplification: float = 10.0) -> float:
    """Stage III rate: secondary phase precipitation causes rate resumption."""
    return r2 * amplification


def boron_release_rate(
    dissolution_rate_m_s: float | np.ndarray,
    boron_mol_fraction: float = 0.145,
    molar_volume_m3_mol: float = 2.4e-5,
) -> float | np.ndarray:
    """Boron normalised mass loss rate [mol m^-2 s^-1]."""
    return np.asarray(dissolution_rate_m_s) * boron_mol_fraction / molar_volume_m3_mol


def compute_bnl(
    boron_release_mol_m2_s: float | np.ndarray,
    time_s: float | np.ndarray,
    surface_area_m2: float = 1.0,
    solution_volume_m3: float = 1.0e-3,
) -> np.ndarray:
    """Cumulative Boron Normalised Loss [g/m^2]. BNL = integral(r_B * dt) * M_B."""
    M_B = 10.81  # g/mol
    r = np.asarray(boron_release_mol_m2_s)
    t = np.asarray(time_s)
    if r.ndim == 0:
        return float(r) * float(t) * M_B
    return np.trapz(r, t) * M_B
