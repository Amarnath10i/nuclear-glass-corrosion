"""
Glass dissolution thermodynamics.

Implements Transition-State Theory (TST) affinity-based rate law
for borosilicate glass corrosion.

References:
    Grambow, B. (1985). MRS Symp. Proc. 44, 15-27.
    Gin, S. et al. (2013). Mater. Today 16, 243-248.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R_GAS: float = 8.314462618  # J mol^-1 K^-1


@dataclass
class GlassComposition:
    """Simplified borosilicate glass composition (mol fractions)."""

    SiO2: float = 0.545
    B2O3: float = 0.145
    Na2O: float = 0.090
    Al2O3: float = 0.060
    CaO: float = 0.050
    ZrO2: float = 0.020
    other: float = 0.090

    def __post_init__(self) -> None:
        total = sum(vars(self).values())
        if not math.isclose(total, 1.0, abs_tol=0.01):
            raise ValueError(f"Mol fractions must sum to 1.0, got {total:.3f}")


def arrhenius_rate_constant(
    k0: float, Ea_kJ_mol: float, temperature_K: float | np.ndarray
) -> float | np.ndarray:
    """Compute Arrhenius rate constant: r_0 = k0 * exp(-Ea / RT)."""
    Ea_J_mol = Ea_kJ_mol * 1e3
    return k0 * np.exp(-Ea_J_mol / (R_GAS * np.asarray(temperature_K, dtype=float)))


def affinity_factor(
    SI: float | np.ndarray,
    temperature_K: float | np.ndarray,
    eta: float = 1.0,
) -> float | np.ndarray:
    """Thermodynamic affinity factor (1 - exp(-A/RT))."""
    RT = R_GAS * temperature_K
    A = -2.303 * RT * np.asarray(SI, dtype=float)
    return (1.0 - np.exp(-A / RT)) ** eta


def dissolution_rate(
    k0: float,
    Ea_kJ_mol: float,
    temperature_K: float | np.ndarray,
    pH: float | np.ndarray,
    SI: float | np.ndarray,
    eta: float = 1.0,
    pH_power: float = -0.4,
) -> float | np.ndarray:
    """Full TST dissolution rate [mol m^-2 s^-1]: r = k(T) * a(H+)^n * f(A)."""
    k = arrhenius_rate_constant(k0, Ea_kJ_mol, temperature_K)
    aH = 10.0 ** (-pH)
    f_aff = affinity_factor(SI, temperature_K, eta)
    return np.asarray(k * (aH ** pH_power) * f_aff)


def gel_layer_thickness(
    rate_m_s: float | np.ndarray,
    time_s: float | np.ndarray,
    density_kg_m3: float = 2500.0,
    molar_mass_kg_mol: float = 0.060,
) -> float | np.ndarray:
    """Parabolic gel layer growth law [m]."""
    rate_mol = np.asarray(rate_m_s) * density_kg_m3 / molar_mass_kg_mol
    return np.asarray(np.sqrt(2.0 * rate_mol * np.asarray(time_s)))
