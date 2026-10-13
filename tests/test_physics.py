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


class TestGelLayerThickness:
    def test_parabolic_law_matches_its_ode(self):
        from nuclear_glass.physics.thermodynamics import gel_layer_thickness

        d, t = 1e-18, np.linspace(1e3, 1e8, 200)
        thickness = gel_layer_thickness(d, t)
        # dL/dt = D / L
        assert np.allclose(np.gradient(thickness, t)[5:-5], d / thickness[5:-5], rtol=1e-2)

    def test_units_are_metres(self):
        from nuclear_glass.physics.thermodynamics import gel_layer_thickness

        # sqrt(2 * 1e-18 m^2/s * 1e8 s) = sqrt(2e-10) m ~ 14 micrometres
        assert math.isclose(float(gel_layer_thickness(1e-18, 1e8)), math.sqrt(2e-10))


class TestAlterationHistory:
    def test_stage_one_is_linear_and_stage_two_parabolic(self):
        from nuclear_glass.physics.kinetics import alteration_history

        r1, d = 1e-12, 1e-18
        t_c = d / r1**2
        t = np.array([0.5 * t_c, 4 * t_c, 100 * t_c])
        thickness, _ = alteration_history(t, r1, d)
        assert math.isclose(thickness[0], r1 * 0.5 * t_c)
        l_c = d / r1
        assert np.allclose(thickness[1:], np.sqrt(l_c**2 + 2 * d * (t[1:] - t_c)))

    def test_continuous_monotone_and_resumption_speeds_it_up(self):
        from nuclear_glass.physics.kinetics import alteration_history

        t = np.logspace(0, 12, 2000)
        base, _ = alteration_history(t, 1e-12, 1e-18)
        res, bnl = alteration_history(t, 1e-12, 1e-18, resumption_time_s=1e9, amplification=10.0)
        assert np.all(np.diff(base) >= 0) and np.all(np.diff(res) >= 0)
        t_c = 1e-18 / 1e-12**2
        for t_switch, kw in ((t_c, {}), (1e9, {"resumption_time_s": 1e9})):
            near, _ = alteration_history(np.array([t_switch * (1 - 1e-9), t_switch * (1 + 1e-9)]), 1e-12, 1e-18, **kw)
            assert math.isclose(near[0], near[1], rel_tol=1e-6)  # continuous at each stage change
        assert np.all(res[t > 1e9] > base[t > 1e9]) and np.allclose(res[t <= 1e9], base[t <= 1e9])
        assert np.allclose(bnl, 2.5e6 * res)
