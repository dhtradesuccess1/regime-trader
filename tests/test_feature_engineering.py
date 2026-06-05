"""Tests for ``core.feature_engineering``.

Covers: the exact set of output columns, the absence of scaling/z-scoring,
removal of the 20-row warmup period, the ValueError raised on missing input
columns, and the warning emitted when too many output rows contain NaN.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from core.feature_engineering import (
    FEATURE_COLUMNS,
    WARMUP_ROWS,
    compute_features,
)


def make_ohlcv(n_rows: int = 120, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic but well-formed OHLCV frame for testing."""
    rng = np.random.default_rng(seed)
    # Random-walk close around 400 (SPY-ish levels).
    returns = rng.normal(0.0005, 0.01, size=n_rows)
    close = 400 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0.003, 0.002, size=n_rows)))
    low = close * (1 - np.abs(rng.normal(0.003, 0.002, size=n_rows)))
    open_ = close * (1 + rng.normal(0, 0.002, size=n_rows))
    volume = rng.integers(50_000_000, 120_000_000, size=n_rows).astype(float)
    index = pd.date_range("2023-01-02", periods=n_rows, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def test_output_columns():
    """Exactly the five feature columns are returned, in order."""
    out = compute_features(make_ohlcv())
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(out.columns) == 5


def test_no_scaling():
    """Features are raw, not z-scored: column means are not all ~0."""
    out = compute_features(make_ohlcv())
    means = out.mean()
    # Volatility and volume_ratio are strictly positive, so a z-scored frame
    # (every column mean ~0) is impossible here.
    assert means["volume_ratio"] == pytest.approx(1.0, abs=0.5)
    assert means["realized_vol_20d"] > 0.01
    # No column should look standardized (mean ~0 AND std ~1 together).
    assert not all(abs(means) < 1e-6)


def test_warmup_rows_dropped():
    """The first 20 input rows do not appear in the output index."""
    df = make_ohlcv()
    out = compute_features(df)
    dropped_index = df.index[:WARMUP_ROWS]
    assert not out.index.isin(dropped_index).any()
    assert len(out) == len(df) - WARMUP_ROWS
    # First surviving row is the input row at position WARMUP_ROWS.
    assert out.index[0] == df.index[WARMUP_ROWS]


def test_raises_on_missing_column():
    """A missing required column raises ValueError."""
    df = make_ohlcv().drop(columns=["Volume"])
    with pytest.raises(ValueError, match="Volume"):
        compute_features(df)


def test_nan_handling(caplog):
    """Injecting a block of NaNs triggers the >2% NaN warning."""
    df = make_ohlcv()
    # Blank out Close for a contiguous block well past the warmup window so
    # NaNs survive into the output and exceed the 2% threshold.
    df.loc[df.index[60:75], "Close"] = np.nan
    with caplog.at_level(logging.WARNING, logger="core.feature_engineering"):
        compute_features(df)
    assert any(
        "NaN" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )
