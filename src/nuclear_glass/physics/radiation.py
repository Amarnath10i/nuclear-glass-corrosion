"""
Radiation-enhanced corrosion module.

Models alpha-recoil damage and radiolysis effects on glass dissolution.

References:
    Weber, W.J. et al. (1997). J. Mater. Res. 13, 1434.
    Gin, S. et al. (2020). Npj Mater. Degrad. 4, 6.
"""

from __future__ import annotations

import math


def alpha_recoil_damage_dpa(
    activity_Bq_m3: float,
    time_s: float,
    displacement_threshold_eV: float = 25.0,
    recoil_energy_MeV: float = 0.1,
) -> float:
    """Estimate displacements per atom (dpa) from alpha recoil (Kinchin-Pease)."""
    recoil_energy_eV = recoil_energy_MeV * 1e6
    N_displ = recoil_energy_eV / (2.0 * displacement_threshold_eV)
    fluence = activity_Bq_m3 * time_s
    N_atoms_m3 = 1.0e28  # borosilicate glass
    return fluence * N_displ / N_atoms_m3


def radiolysis_H2O2_concentration(
    dose_rate_Gy_s: float,
    time_s: float,
    G_H2O2: float = 0.07e-6,  # mol/J
) -> float:
    """Estimate H2O2 concentration from water radiolysis."""
    total_dose_J_m3 = dose_rate_Gy_s * time_s * 1e3
    return G_H2O2 * total_dose_J_m3


def radiation_enhanced_rate_factor(
    dpa: float,
    saturation_dpa: float = 0.1,
    max_enhancement: float = 5.0,
) -> float:
    """Empirical rate enhancement: f = 1 + (max-1)*(1-exp(-dpa/dpa_sat))."""
    return 1.0 + (max_enhancement - 1.0) * (1.0 - math.exp(-dpa / saturation_dpa))


def cumulative_dose_from_inventory(
    activity_Bq: float,
    time_s: float,
    half_life_s: float,
    alpha_energy_MeV: float = 5.0,
    glass_mass_kg: float = 1.0,
) -> float:
    """Integrate cumulative alpha dose [Gy] over radioactive decay."""
    lam = math.log(2) / half_life_s
    n_decays = (activity_Bq / lam) * (1.0 - math.exp(-lam * time_s))
    energy_J = n_decays * alpha_energy_MeV * 1.602e-13
    return energy_J / glass_mass_kg
