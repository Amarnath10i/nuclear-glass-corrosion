"""Unit tests for physics modules."""

import math

import numpy as np
import pytest

from nuclear_glass.physics.kinetics import boron_release_rate, stage2_residual_rate
from nuclear_glass.physics.thermodynamics import (
    GlassComposition,
    affinity_factor,
    arrhenius_rate_constant,
)
from nuclear_glass.physics.transport import silicic_acid_saturation


class TestArrheniusRateConstant:
    def test_positive_output(self):
        assert arrhenius_rate_constant(1e-10, 72.0, 298.15) > 0

    def test_increases_with_temperature(self):
        k1 = arrhenius_rate_constant(1e-10, 72.0, 298.15)
        k2 = arrhenius_rate_constant(1e-10, 72.0, 363.15)
        assert k2 > k1

    def test_zero_ea_gives_k0(self):
        assert math.isclose(arrhenius_rate_constant(1e-10, 0.0, 298.15), 1e-10, rel_tol=1e-6)


class TestAffinityFactor:
    def test_zero_si_gives_zero_rate(self):
        assert math.isclose(float(affinity_factor(0.0, 298.15)), 0.0, abs_tol=1e-9)

    def test_very_negative_si_gives_near_one(self):
        assert float(affinity_factor(-10.0, 298.15)) > 0.99

    def test_array_decreases_toward_equilibrium(self):
        SI = np.array([-2.0, -1.0, 0.0])
        f = affinity_factor(SI, 298.15)
        assert np.all(np.diff(f) < 0)


class TestGlassComposition:
    def test_valid_composition(self):
        gc = GlassComposition()
        assert math.isclose(sum(vars(gc).values()), 1.0, abs_tol=0.01)

    def test_invalid_composition_raises(self):
        with pytest.raises(ValueError):
            GlassComposition(SiO2=0.9, B2O3=0.9)


class TestKinetics:
    def test_stage2_decreases_with_gel_thickness(self):
        assert stage2_residual_rate(1e-12, 1e-6) > stage2_residual_rate(1e-12, 1e-5)

    def test_zero_gel_returns_stage1(self):
        assert stage2_residual_rate(1e-12, 0.0) == 1e-12

    def test_boron_release_positive(self):
        assert float(boron_release_rate(1e-9)) > 0


class TestTransport:
    def test_sat_index_at_ksp(self):
        T = 298.15
        log_Ksp = -0.338 - 7.889e-4 * T
        Ksp_mol_m3 = 10**log_Ksp * 1e3
        SI = silicic_acid_saturation(Ksp_mol_m3, T)
        assert abs(float(SI)) < 0.01
