"""
Experimental data loaders for standard glass corrosion datasets.

Supported datasets:
  - SRL 165 (Savannah River Lab, PCT-A tests)
  - ISG (International Simple Glass)
  - MCC-1 static leach tests
  - VHT (Vapor Hydration Tests)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class GlassCorrosionDataset:
    """Base class for glass corrosion experimental datasets."""

    REQUIRED_COLUMNS = ["time_days", "temperature_C", "pH", "BNL_g_m2"]

    def __init__(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.df = df.copy()
        self.df["time_s"] = self.df["time_days"] * 86_400.0
        self.df["temperature_K"] = self.df["temperature_C"] + 273.15

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n={len(self)}, "
            f"T={self.df['temperature_C'].unique().tolist()}C)"
        )

    def normalised_loss_rate(self) -> pd.Series:
        """Compute BNL rate [g m^-2 day^-1] by finite differences."""
        return self.df["BNL_g_m2"].diff() / self.df["time_days"].diff()


def load_srl165(path: Path | str) -> GlassCorrosionDataset:
    """Load SRL 165 glass PCT-A dataset."""
    df = pd.read_csv(path)
    return GlassCorrosionDataset(df)


def load_isg(path: Path | str) -> GlassCorrosionDataset:
    """Load International Simple Glass (ISG) dataset."""
    df = pd.read_csv(path)
    if "NL_B_g_m2" in df.columns and "BNL_g_m2" not in df.columns:
        df = df.rename(columns={"NL_B_g_m2": "BNL_g_m2"})
    return GlassCorrosionDataset(df)


def load_vht(path: Path | str, temperature_C: float = 200.0) -> GlassCorrosionDataset:
    """Load Vapor Hydration Test (accelerated) data."""
    df = pd.read_csv(path)
    if "temperature_C" not in df.columns:
        df["temperature_C"] = temperature_C
    return GlassCorrosionDataset(df)


def synthetic_srl165(n_samples: int = 500, seed: int = 42) -> GlassCorrosionDataset:
    """Generate synthetic SRL 165-like data for unit tests."""
    from nuclear_glass.physics.thermodynamics import dissolution_rate

    rng = np.random.default_rng(seed)
    times_days = np.sort(rng.uniform(1, 365 * 5, n_samples))
    temps_C    = rng.choice([25, 40, 70, 90], n_samples)
    pHs        = rng.uniform(6.5, 9.5, n_samples)
    SIs        = rng.uniform(-2.0, 0.0, n_samples)

    rates = dissolution_rate(
        k0=1e-10,
        Ea_kJ_mol=72.0,
        temperature_K=temps_C + 273.15,
        pH=pHs,
        SI=SIs,
    )
    bnl = rates * times_days * 86400.0 * 10.81 * 0.145 / 2.4e-5 * 1e-3
    noise = rng.normal(0, 0.05 * bnl.mean(), n_samples)

    df = pd.DataFrame({
        "time_days": times_days,
        "temperature_C": temps_C.astype(float),
        "pH": pHs,
        "BNL_g_m2": bnl + noise,
    })
    return GlassCorrosionDataset(df)
