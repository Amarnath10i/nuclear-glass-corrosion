"""Unit tests for data loaders."""

import pandas as pd
import pytest

from nuclear_glass.data.loaders import GlassCorrosionDataset, synthetic_srl165


def test_synthetic_dataset_length():
    assert len(synthetic_srl165(n_samples=100)) == 100


def test_synthetic_dataset_columns():
    ds = synthetic_srl165(n_samples=50)
    assert "time_days" in ds.df.columns
    assert "BNL_g_m2" in ds.df.columns
    assert "temperature_K" in ds.df.columns


def test_bnl_non_negative_mean():
    assert synthetic_srl165(n_samples=200).df["BNL_g_m2"].mean() > 0


def test_missing_column_raises():
    df = pd.DataFrame({"time_days": [1, 2], "temperature_C": [25, 25]})
    with pytest.raises(ValueError):
        GlassCorrosionDataset(df)
