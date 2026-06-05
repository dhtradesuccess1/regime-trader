"""Tests for ``data.data_validator``.

Covers detection of missing columns, NaN rows, non-positive volume and prices,
duplicate / non-monotonic timestamps, calendar gaps, and the clean-data
happy path.
"""

import numpy as np
import pandas as pd

from data.data_validator import is_valid, validate_ohlcv


def make_clean(n_rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 400 + np.cumsum(rng.normal(0, 1, size=n_rows))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n_rows).astype(float),
        },
        index=pd.date_range("2023-01-02", periods=n_rows, freq="B"),
    )


def test_clean_data_is_valid():
    report = validate_ohlcv(make_clean(), ticker="SPY")
    assert report["is_valid"] is True
    assert report["issues"] == []
    assert is_valid(make_clean(), "SPY") is True


def test_missing_column():
    df = make_clean().drop(columns=["Volume"])
    report = validate_ohlcv(df, "SPY")
    assert report["is_valid"] is False
    assert "Volume" in report["missing_columns"]


def test_nan_detected():
    df = make_clean()
    df.loc[df.index[5], "Close"] = np.nan
    report = validate_ohlcv(df)
    assert report["is_valid"] is False
    assert report["nan_rows"] >= 1


def test_nonpositive_volume_detected():
    df = make_clean()
    df.loc[df.index[3], "Volume"] = 0
    report = validate_ohlcv(df)
    assert report["is_valid"] is False
    assert report["nonpositive_volume"] >= 1


def test_nonpositive_price_detected():
    df = make_clean()
    df.loc[df.index[4], "Low"] = -1.0
    report = validate_ohlcv(df)
    assert report["is_valid"] is False
    assert report["nonpositive_price"] >= 1


def test_gap_detected():
    df = make_clean(30)
    # Drop a two-week chunk to create a calendar gap.
    df = pd.concat([df.iloc[:10], df.iloc[20:]])
    report = validate_ohlcv(df)
    assert report["gap_count"] >= 1


def test_non_monotonic_detected():
    df = make_clean(20)
    df = df.iloc[::-1]  # reverse -> not monotonic increasing
    report = validate_ohlcv(df)
    assert report["is_valid"] is False
    assert report["monotonic"] is False
