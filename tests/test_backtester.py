"""Tests for ``core.backtester``.

Covers the cost model (regime/size-dependent slippage, partial fills, gap
blocking), walk-forward correctness (no look-ahead, minimum-window
enforcement), and the presence of all required benchmark outputs.
"""

import numpy as np
import pandas as pd
import pytest

from core.backtester import (
    WalkForwardBacktester,
    calculate_slippage,
    check_gap_open,
    simulate_fill_rate,
)
from core.feature_engineering import compute_features
from core.hmm_engine import HMMRegimeEngine


def make_ohlcv(n_rows: int, seed: int = 5) -> pd.DataFrame:
    """Synthetic OHLCV with two regimes so the HMM has structure to find."""
    rng = np.random.default_rng(seed)
    half = n_rows // 2
    calm = rng.normal(0.0007, 0.007, size=half)
    stormy = rng.normal(-0.0010, 0.020, size=n_rows - half)
    returns = np.concatenate([calm, stormy])
    close = 400 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0.003, 0.002, size=n_rows)))
    low = close * (1 - np.abs(rng.normal(0.003, 0.002, size=n_rows)))
    open_ = close * (1 + rng.normal(0, 0.002, size=n_rows))
    volume = rng.integers(50_000_000, 120_000_000, size=n_rows).astype(float)
    index = pd.date_range("2018-01-02", periods=n_rows, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


# --------------------------------------------------------------- cost model
def test_crash_slippage_exceeds_bull():
    assert calculate_slippage("crash", 0.001) > calculate_slippage("bull", 0.001)


def test_partial_fills_in_bear():
    np.random.seed(0)
    rates = [simulate_fill_rate(0.005, "bear") for _ in range(50)]
    assert np.mean(rates) < 0.95


def test_fill_rate_full_for_tiny_orders():
    assert simulate_fill_rate(0.0005, "crash") == 1.0


def test_gap_open_blocks_entry():
    """A 3% gap blocks a new entry: target weight cannot rise above current."""
    bt = WalkForwardBacktester(make_ohlcv(400))
    prev_close = 100.0
    gap_open = 103.0  # +3% gap, exceeds the 2% threshold
    assert check_gap_open(gap_open, prev_close) is True
    # Strategy wants full exposure (1.0) from flat (0.0); the gap blocks it.
    result = bt.apply_gap_filter(
        target_weight=1.0, current_weight=0.0, open_price=gap_open, prev_close=prev_close
    )
    assert result == 0.0  # no new position opened
    # A non-gap session would have allowed the entry.
    assert bt.apply_gap_filter(1.0, 0.0, 100.5, prev_close) == 1.0


def test_gap_open_reduces_existing_by_half():
    """On a gap, an existing position is cut to 50% even if the target is full."""
    bt = WalkForwardBacktester(make_ohlcv(400))
    prev_close, gap_open = 100.0, 103.0  # +3% gap
    # Strategy still wants full exposure (1.0) but currently holds 0.8.
    result = bt.apply_gap_filter(
        target_weight=1.0, current_weight=0.8, open_price=gap_open, prev_close=prev_close
    )
    assert result == pytest.approx(0.4)  # 0.8 halved, increase blocked
    # Without a gap the position is free to move to target.
    assert bt.apply_gap_filter(1.0, 0.8, 100.5, prev_close) == 1.0


# ----------------------------------------------------------- no look-ahead
def test_no_future_data():
    """Decisions for bars 0..N use only data up to each bar.

    Mutating raw prices in the tail must not change any decision that precedes
    the mutated region.
    """
    data = make_ohlcv(160, seed=3)
    engine = HMMRegimeEngine((3, 4)).fit(compute_features(make_ohlcv(300, seed=9)))
    bt = WalkForwardBacktester(data, n_regimes_range=(3, 4))

    base = bt.simulate_window(engine, data, window_index=0)

    # Corrupt the last 10 raw bars; features depend only on trailing data, so
    # feature rows at price positions < cutoff are untouched.
    cutoff = len(data) - 10
    mutated = data.copy()
    mutated.iloc[cutoff:, :] *= 1.5
    after = bt.simulate_window(engine, mutated, window_index=0)

    # Feature index j maps to price position j + 20 (warmup dropped); compare
    # the prefix guaranteed unaffected by the mutation.
    safe_len = cutoff - 20
    assert safe_len > 5
    assert base["regimes"][:safe_len] == after["regimes"][:safe_len]
    assert base["weights"][:safe_len] == pytest.approx(after["weights"][:safe_len])


# ------------------------------------------------------------ walk-forward
def test_minimum_windows_enforced():
    """Fewer than 4 windows of data raises ValueError."""
    # train+oos = 315; with only 330 bars only 1 window fits.
    short = make_ohlcv(330)
    bt = WalkForwardBacktester(short)
    with pytest.raises(ValueError, match="window"):
        bt.run()


def test_all_benchmarks_present():
    """A real walk-forward run yields all three benchmark keys per window."""
    # Enough bars for >= 4 windows: train 252 + oos 63 + 3*step 63 = 378.
    data = make_ohlcv(450)
    bt = WalkForwardBacktester(data, n_regimes_range=(3, 4))
    results = bt.run()

    assert len(results) >= 4
    required = {
        "total_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "avg_hold_bars",
        "total_trades",
        "regime_breakdown",
        "avg_slippage_bps",
        "avg_fill_rate",
        "benchmark_buy_hold",
        "benchmark_sma200",
        "benchmark_random_median",
    }
    for window in results:
        assert required.issubset(window.keys())
        assert isinstance(window["regime_breakdown"], dict)
