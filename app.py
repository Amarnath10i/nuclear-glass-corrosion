"""Interactive explorer for the glass-corrosion physics and the conformal-prediction module.

    streamlit run app.py

NumPy/pandas only, so it also runs in the browser (Pyodide/stlite).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nuclear_glass.data.loaders import synthetic_srl165  # noqa: E402
from nuclear_glass.models.conformal import SplitConformalPredictor  # noqa: E402
from nuclear_glass.physics.kinetics import alteration_history  # noqa: E402
from nuclear_glass.physics.thermodynamics import affinity_factor, dissolution_rate  # noqa: E402

st.set_page_config(page_title="Glass corrosion explorer", page_icon="🧪", layout="wide")
st.title("Nuclear waste glass corrosion: physics explorer")
st.caption("Illustrative models with adjustable, uncalibrated parameters. Nothing here is fitted to laboratory "
           "data; see the README for what the project does and does not do yet.")

with st.sidebar:
    st.header("Rate law")
    log_k0 = st.slider("log10 k0 (mol/m²/s)", -3.0, 3.0, 0.0, 0.1)
    ea = st.slider("Activation energy Ea (kJ/mol)", 40.0, 100.0, 72.0, 1.0)
    eta = st.slider("Affinity exponent η", 0.5, 2.0, 1.0, 0.1)
    st.caption("r = k0 · exp(−Ea/RT) · a(H⁺)^−0.4 · (1 − 10^SI)^η. The defaults give a forward rate of "
               "about 1 g/m²/day at 90 °C and pH 9, the order of magnitude reported for borosilicate glasses.")

tab_rate, tab_time, tab_cp = st.tabs(["Dissolution rate", "Long-term alteration", "Conformal intervals"])

with tab_rate:
    T = np.linspace(20, 100, 200) + 273.15
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    for ph in (7, 8, 9, 10):
        ax[0].semilogy(T - 273.15, dissolution_rate(10 ** log_k0, ea, T, ph, -3.0, eta), label=f"pH {ph}")
    ax[0].set(xlabel="temperature (°C)", ylabel="forward rate", title="Temperature and pH (far from saturation)")
    ax[0].legend()
    ax[1].semilogy(1000 / T, dissolution_rate(10 ** log_k0, ea, T, 9.0, -3.0, eta))
    ax[1].set(xlabel="1000 / T (1/K)", title=f"Arrhenius plot: slope = −Ea/R ({ea:.0f} kJ/mol)")
    si = np.linspace(-3, 0, 200)
    for e in (0.5, 1.0, 2.0):
        ax[2].plot(si, affinity_factor(si, 363.15, e), label=f"η = {e}")
    ax[2].set(xlabel="saturation index SI of silica", ylabel="(1 − 10^SI)^η",
              title="Affinity: the rate drops as the solution saturates")
    ax[2].legend()
    fig.tight_layout()
    st.pyplot(fig)

with tab_time:
    c1, c2, c3 = st.columns(3)
    temp = c1.slider("Repository temperature (°C)", 20, 90, 50)
    ph = c1.slider("pH", 7.0, 10.0, 9.0, 0.1)
    log_d = c2.slider("log10 gel diffusivity D (m²/s)", -22.0, -16.0, -19.0, 0.1)
    resume = c2.checkbox("Stage III resumption", value=False)
    t_r_years = c3.select_slider("Resumption at (years)", [1, 10, 100, 1000, 10_000], value=1000, disabled=not resume)
    amp = c3.slider("Diffusivity amplification after resumption", 1.0, 100.0, 10.0, disabled=not resume)
    # Forward rate (mol m^-2 s^-1) times the glass molar volume (m^3/mol) = retreat velocity (m/s).
    r1 = float(dissolution_rate(10 ** log_k0, ea, temp + 273.15, ph, -3.0, eta)) * 2.4e-5
    year = 365.25 * 86400
    t = np.logspace(0, np.log10(1e5 * year), 400)
    thick, bnl = alteration_history(t, r1, 10 ** log_d, t_r_years * year if resume else None, amp)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.loglog(t / year, bnl)
    ax.axvline(10 ** log_d / r1 ** 2 / year, ls=":", color="0.5", label="Stage I → II")
    if resume:
        ax.axvline(t_r_years, ls="--", color="C3", label="Stage III")
    ax.set(xlabel="time (years)", ylabel="boron normalised loss NL_B (g/m²)",
           title=f"Altered glass after 100,000 years: {1e3 * thick[-1]:.2f} mm")
    ax.legend()
    st.pyplot(fig)
    st.caption("Stage I: the glass retreats at the forward rate. Stage II: once diffusion through the gel is slower, "
               "growth is parabolic (∝ √t). Stage III: secondary phases destabilise the gel. The 100,000-year number "
               "is only as good as the parameters, which is why the project's goal is uncertainty, not a point value.")

with tab_cp:
    st.markdown("Split conformal prediction gives intervals with **guaranteed** coverage whatever the model, if "
                "calibration and test points are exchangeable. Here a deliberately simple least-squares model "
                "predicts boron loss on synthetic data drawn from the rate law above (with an unobserved "
                "saturation index acting as noise).")
    df = synthetic_srl165(n_samples=1500, seed=0).df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    temp_k = df["temperature_C"].to_numpy() + 273.15
    x = df["time_days"].to_numpy() * np.exp(-72e3 / (8.314 * temp_k)) * 10 ** (0.4 * df["pH"].to_numpy())
    X = np.column_stack([np.ones(len(df)), x / x.std()])
    y = df["BNL_g_m2"].to_numpy()
    tr, cal, te = slice(0, 500), slice(500, 1000), slice(1000, 1500)
    beta = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    pred = X @ beta
    alphas = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    cover, width = [], []
    for a in alphas:
        cp = SplitConformalPredictor(alpha=float(a)).calibrate(y[cal], pred[cal])
        lo, hi = cp.predict(pred[te])
        cover.append(np.mean((y[te] >= lo) & (y[te] <= hi)))
        width.append(cp.interval_width)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(1 - alphas, cover, "o-", label="empirical (500 test points)")
    ax[0].plot([0.5, 1], [0.5, 1], "k--", lw=1, label="target")
    ax[0].set(xlabel="target coverage 1 − α", ylabel="coverage on test points", title="Coverage")
    ax[0].legend()
    order = np.argsort(pred[te])
    cp = SplitConformalPredictor(alpha=0.1).calibrate(y[cal], pred[cal])
    lo, hi = cp.predict(pred[te][order])
    ax[1].scatter(np.arange(500), y[te][order], s=4, color="0.4", label="observed")
    ax[1].fill_between(np.arange(500), lo, hi, alpha=0.3, label="90% conformal interval")
    ax[1].set(xlabel="test points (sorted by prediction)", ylabel="NL_B (g/m²)", title="Intervals")
    ax[1].legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Constant-width intervals from absolute residuals are valid but not adaptive: they are too wide "
               "for small losses and tight for large ones. Conformalised quantile regression "
               "(QuantileConformalPredictor) fixes that.")
