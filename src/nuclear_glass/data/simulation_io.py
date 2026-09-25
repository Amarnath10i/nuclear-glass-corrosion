"""PFLOTRAN / TOUGHREACT simulation output parsers for multi-fidelity learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_pflotran_h5(path: Path | str, group: str = "/Observation") -> pd.DataFrame:
    """Parse PFLOTRAN HDF5 output into a tidy DataFrame."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required: pip install h5py") from e

    rows: list[dict[str, Any]] = []
    with h5py.File(path, "r") as f:
        grp = f[group]
        times = grp["Time [s]"][:]
        for i, t in enumerate(times):
            row: dict[str, Any] = {"time_s": float(t)}
            for key in grp:
                if key == "Time [s]":
                    continue
                data = grp[key][:]
                row[key.replace(" ", "_")] = float(data[i]) if data.ndim == 1 else data[i].tolist()
            rows.append(row)
    return pd.DataFrame(rows)


def parse_toughreact_csv(path: Path | str) -> pd.DataFrame:
    """Parse TOUGHREACT tabular output."""
    df = pd.read_csv(path, comment="#", sep=r"\s+", engine="python")
    df.columns = [
        c.strip().lower().replace("(", "_").replace(")", "").replace(" ", "_")
        for c in df.columns
    ]
    return df


def interpolate_to_common_grid(
    dfs: list[pd.DataFrame],
    time_col: str = "time_s",
    n_points: int = 200,
) -> pd.DataFrame:
    """Interpolate multiple simulation outputs to a common time grid."""
    import scipy.interpolate as interp

    t_min = max(df[time_col].min() for df in dfs)
    t_max = min(df[time_col].max() for df in dfs)
    t_common = np.linspace(t_min, t_max, n_points)

    result_parts: list[pd.DataFrame] = []
    for i, df in enumerate(dfs):
        row: dict[str, np.ndarray] = {time_col: t_common}
        for col in df.select_dtypes(include="number").columns:
            if col == time_col:
                continue
            f = interp.interp1d(df[time_col].values, df[col].values, fill_value="extrapolate")
            row[col] = f(t_common)
        part = pd.DataFrame(row)
        part["fidelity_level"] = i
        result_parts.append(part)

    return pd.concat(result_parts, ignore_index=True)
