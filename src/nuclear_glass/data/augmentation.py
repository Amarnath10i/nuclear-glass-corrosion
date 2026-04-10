"""
Physics-aware data augmentation for glass corrosion datasets.

Strategies:
  1. Gaussian noise on observations
  2. Temperature perturbation via Arrhenius scaling
  3. Time-series interpolation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def gaussian_noise_augment(
    df: pd.DataFrame,
    noise_fraction: float = 0.05,
    seed: int = 0,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add Gaussian noise to numeric measurement columns."""
    rng = np.random.default_rng(seed)
    df_aug = df.copy()
    if cols is None:
        cols = [
            c for c in df.select_dtypes(include="number").columns
            if not c.startswith("time_")
        ]
    for col in cols:
        sigma = noise_fraction * df[col].abs().mean()
        df_aug[col] = df[col] + rng.normal(0, sigma, len(df))
        if col in ("BNL_g_m2", "pH"):
            df_aug[col] = df_aug[col].clip(lower=0.0)
    return df_aug


def arrhenius_temperature_augment(
    df: pd.DataFrame,
    Ea_kJ_mol: float = 72.0,
    delta_T_K: float = 5.0,
    seed: int = 1,
) -> pd.DataFrame:
    """Scale BNL via Arrhenius law for small temperature perturbations."""
    R = 8.314
    rng = np.random.default_rng(seed)
    df_aug = df.copy()
    T_old = df["temperature_K"].values
    dT = rng.uniform(-delta_T_K, delta_T_K, len(df))
    T_new = T_old + dT
    scale = np.exp(-Ea_kJ_mol * 1e3 / R * (1.0 / T_new - 1.0 / T_old))
    df_aug["temperature_K"] = T_new
    df_aug["temperature_C"] = T_new - 273.15
    df_aug["BNL_g_m2"] = df["BNL_g_m2"] * scale
    return df_aug


def interpolate_time_series(
    df: pd.DataFrame,
    factor: int = 4,
    method: str = "linear",
) -> pd.DataFrame:
    """Densify time axis by interpolation."""
    import scipy.interpolate as interp

    df = df.sort_values("time_days").reset_index(drop=True)
    t_old = df["time_days"].values
    t_new = np.linspace(t_old[0], t_old[-1], len(t_old) * factor)
    out: dict[str, np.ndarray] = {"time_days": t_new}

    for col in df.select_dtypes(include="number").columns:
        if col == "time_days":
            continue
        f = interp.interp1d(t_old, df[col].values, kind=method, fill_value="extrapolate")
        out[col] = f(t_new)

    return pd.DataFrame(out)
