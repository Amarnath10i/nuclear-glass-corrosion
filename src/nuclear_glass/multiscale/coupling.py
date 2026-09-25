"""Multi-scale coupling: atomistic MD to mesoscale to continuum."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class AtomisticOutput:
    activation_energy_eV: float
    pre_exponential_s: float
    surface_energy_J_m2: float
    diffusivity_m2_s: float


@dataclass
class MesoscaleOutput:
    gel_thickness_m: float
    porosity: float
    tortuosity: float


@dataclass
class ContinuumInput:
    effective_diffusivity_m2_s: float
    surface_area_m2_m3: float
    dissolution_rate_m_s: float


def upscale_atomistic_to_mesoscale(atomistic: AtomisticOutput, temperature_K: float = 298.15) -> dict[str, float]:
    R = 8.314
    Ea = atomistic.activation_energy_eV * 96485.0
    rate = atomistic.pre_exponential_s * np.exp(-Ea / (R * temperature_K))
    return {"nucleation_rate_m2_s": rate, "surface_energy_J_m2": atomistic.surface_energy_J_m2}


def upscale_mesoscale_to_continuum(meso: MesoscaleOutput, D0_m2_s: float = 1.0e-22) -> ContinuumInput:
    """Millington-Quirk effective diffusivity + SSA estimate."""
    phi = meso.porosity
    D_eff = D0_m2_s * (phi ** (10 / 3)) / (phi**2)
    SSA = 1.0 / max(meso.gel_thickness_m, 1e-12)
    return ContinuumInput(
        effective_diffusivity_m2_s=D_eff,
        surface_area_m2_m3=SSA,
        dissolution_rate_m_s=D_eff / max(meso.gel_thickness_m, 1e-12),
    )


def hierarchical_prediction(
    atomistic: AtomisticOutput,
    temperature_K: float,
    time_horizon_s: float,
    mesoscale_surrogate: Callable | None = None,
) -> ContinuumInput:
    meso_inputs = upscale_atomistic_to_mesoscale(atomistic, temperature_K)
    if mesoscale_surrogate is not None:
        meso_out = mesoscale_surrogate(meso_inputs, time_horizon_s)
    else:
        gel_thickness = np.sqrt(2.0 * atomistic.diffusivity_m2_s * time_horizon_s)
        meso_out = MesoscaleOutput(gel_thickness_m=gel_thickness, porosity=0.3, tortuosity=2.5)
    return upscale_mesoscale_to_continuum(meso_out, D0_m2_s=atomistic.diffusivity_m2_s)
