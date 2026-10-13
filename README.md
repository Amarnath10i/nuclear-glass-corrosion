# Nuclear Waste Glass Corrosion

Building blocks for predicting how borosilicate nuclear waste glass corrodes over geological time: glass dissolution physics, physics-informed neural operators, and conformal prediction intervals. Regulators ask for safety cases over ~100,000 years; laboratory tests last years at most, so any long-term prediction is an extrapolation and its uncertainty is the real deliverable.

> **Status: early-stage library, not a validated model.** The physics and conformal-prediction modules are implemented and tested, and there is an interactive explorer. The neural models have implementations but have **not been trained**: the repository contains no laboratory data (loaders for SRL-165, ISG and VHT CSV files exist, but the data is not included), so there are no accuracy results yet, and nothing here is calibrated to real glass.

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The explorer (NumPy only; it also runs in the browser) shows:
- **Dissolution rate**: the transition-state-theory rate law against temperature, pH and silica saturation, with an Arrhenius plot.
- **Long-term alteration**: altered glass thickness and boron loss from 1 second to 100,000 years through three stages: forward dissolution, diffusion-limited growth through the gel (∝ √t), and an optional Stage III resumption.
- **Conformal intervals**: split conformal prediction on synthetic data drawn from the rate law, with empirical coverage against the target at several α.

## What is implemented

| Module | Contents | Tested |
|---|---|---|
| `physics/thermodynamics.py` | Arrhenius rate constant, affinity factor (1 − 10^SI)^η, full TST rate r = k0·exp(−Ea/RT)·a(H⁺)^−0.4·f(A), parabolic gel layer L = √(2Dt) | yes |
| `physics/kinetics.py` | Stage I/II/III rates, boron release, boron normalised loss, closed-form three-stage alteration history | yes |
| `physics/transport.py` | 1D diffusion matrix and transport solve, silica saturation | partly |
| `physics/radiation.py` | alpha-recoil damage, radiolysis, dose-rate enhancement factor | no |
| `models/conformal.py` | split and quantile (CQR) conformal predictors | split: yes, including coverage over repeated splits; CQR: no |
| `models/pino.py`, `deeponet.py`, `bayesian_nn.py` | 1D physics-informed FNO, DeepONet, Bayes-by-backprop MLP (PyTorch) | no; never trained |
| `data/loaders.py` | CSV loaders for SRL-165 / ISG / VHT-style tables; a synthetic generator | yes |
| `evaluation/` | relative L2, extrapolation bias, coverage and sharpness metrics | no |
| `multiscale/` | homogenisation bounds (Hashin–Shtrikman, Kozeny–Carman) and placeholder scale-bridging functions | no |

Fixes made while reviewing the code:
- **Gel layer thickness** returned √(mol·m⁻²), not metres; it is now the diffusion-limited law L = √(2Dt), tested against its ODE.
- **Conformal quantile** crashed for small calibration sets (the quantile level exceeded 1); it now returns an infinite interval, as the method prescribes.
- **Entry points** for `nuclear_glass.cli` pointed at a module that does not exist; they were removed.

## What it would take to be a real model

1. **Data.** Published static-dissolution series (e.g. ISG at 90 °C, SRL-165 PCT/MCC-1) digitised into the loaders' CSV format, with the test conditions (S/V, T, pH) attached.
2. **Calibration.** Bayesian calibration of k0, Ea, η and the gel diffusivity against those series, with held-out experiments for validation.
3. **Surrogate.** Train the neural operators on reactive-transport simulations over the calibrated parameter posterior, so the surrogate is checked against the simulator, not against itself.
4. **Uncertainty.** Propagate parameter uncertainty to 100,000 years and report intervals, with conformal calibration on held-out experiments as a check.

## Install

```bash
pip install -r requirements.txt        # physics, conformal prediction, app
pip install -r requirements-ml.txt     # adds PyTorch for the neural models
pip install -e ".[dev]" && pytest      # tests, ruff and mypy as in CI
```

## Layout

```
app.py                     Streamlit explorer
src/nuclear_glass/
  physics/                 rate laws, staged alteration, transport, radiation
  models/                  conformal prediction; PINO, DeepONet, Bayesian NN (untrained)
  data/                    loaders and synthetic data
  evaluation/              metrics and uncertainty diagnostics
  multiscale/              homogenisation and scale bridging
  training/, utils/        loss functions, trainer, config, plotting
tests/                     physics, conformal and data tests
docs/physics.md            the equations
```

## References

- Grambow, B. (1985). A general rate equation for nuclear waste glass corrosion. *MRS Proceedings* 44.
- Gin, S. et al. (2013). An international initiative on long-term behavior of high-level nuclear waste glass. *Materials Today* 16(6).
- Angelopoulos, A. N. and Bates, S. (2021). A gentle introduction to conformal prediction. arXiv:2107.07511.
