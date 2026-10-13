"""
Dissolution kinetics module.

Implements:
  - Stage I (initial dissolution) - far from saturation
  - Stage II (residual rate) - passivated gel layer regime
  - Stage III (resumption) - alteration layer transformation
"""

from __future__ import annotations

import numpy as np


def stage1_rate(
    k0: float, temperature_K: float | np.ndarray, Ea_kJ_mol: float, pH: float
) -> float | np.ndarray:
    """Forward dissolution rate in Stage I (far from saturation) [m/s]."""
    from nuclear_glass.physics.thermodynamics import arrhenius_rate_constant
    k = arrhenius_rate_constant(k0, Ea_kJ_mol, temperature_K)
    return np.asarray(k * (10.0 ** (-pH)) ** (-0.4))


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
) -> float | np.ndarray:
    """Cumulative Boron Normalised Loss [g/m^2]. BNL = integral(r_B * dt) * M_B."""
    M_B = 10.81  # g/mol
    r = np.asarray(boron_release_mol_m2_s)
    t = np.asarray(time_s)
    if r.ndim == 0:
        return float(r) * float(t) * M_B
    return np.asarray(np.trapezoid(r, t) * M_B)



def alteration_history(
    time_s: np.ndarray,
    forward_rate_m_s: float,
    diffusivity_m2_s: float,
    resumption_time_s: float | None = None,
    amplification: float = 10.0,
    glass_density_g_m3: float = 2.5e6,
) -> tuple[np.ndarray, np.ndarray]:
    """Altered glass thickness [m] and boron normalised mass loss [g/m^2] over time.

    A three-stage illustration of glass alteration, solved in closed form:

    - Stage I: the glass retreats at the forward rate r1 while the gel layer is
      thin, L = r1 t, until diffusion through the gel becomes slower than r1
      (L_c = D / r1, at t_c = D / r1^2).
    - Stage II: diffusion-limited, dL/dt = D / L, so L^2 = L_c^2 + 2 D (t - t_c).
    - Stage III (optional, from resumption_time_s): secondary phases
      destabilise the gel and the effective diffusivity grows by
      `amplification`: L^2 = L_r^2 + 2 A D (t - t_r).

    Boron is released congruently from the altered glass, so NL_B = rho * L.
    """
    t = np.asarray(time_s, dtype=float)
    r1, d = forward_rate_m_s, diffusivity_m2_s
    t_c = d / r1**2
    l_c = d / r1
    thickness = np.where(t < t_c, r1 * t, np.sqrt(l_c**2 + 2.0 * d * np.maximum(t - t_c, 0.0)))
    if resumption_time_s is not None:
        t_r = max(resumption_time_s, 0.0)
        l_r = r1 * t_r if t_r < t_c else np.sqrt(l_c**2 + 2.0 * d * (t_r - t_c))
        thickness = np.where(t > t_r, np.sqrt(l_r**2 + 2.0 * amplification * d * np.maximum(t - t_r, 0.0)),
                             thickness)
    return thickness, glass_density_g_m3 * thickness
