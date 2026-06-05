"""Tests for ``core.risk_manager`` (and the lockfile gate in order_executor).

Covers each staged circuit-breaker level (soft size reduction, halve+block,
close-weakest, close-all, lockfile), the lockfile order gate, stress-adjusted
correlation, and the position-size cap.

An autouse fixture redirects the lockfile to a temp path so tests never touch
the real project-root ``lockfile.lock``.
"""

import numpy as np
import pandas as pd
import pytest

import broker.order_executor as oe
import core.risk_manager as rm_mod
from core.risk_manager import RiskManager, check_gap_risk, get_effective_correlation
from settings.config import MAX_POSITION_SIZE_PCT


@pytest.fixture(autouse=True)
def isolated_lockfile(tmp_path, monkeypatch):
    """Point the lockfile at a temp file for every test."""
    lock = tmp_path / "lockfile.lock"
    monkeypatch.setattr(rm_mod, "LOCKFILE_PATH", lock)
    return lock


def make_positions():
    return {
        "A": {"notional": 10_000.0, "sector": "broad"},
        "B": {"notional": 10_000.0, "sector": "tech"},
        "C": {"notional": 10_000.0, "sector": "tech"},
    }


# ----------------------------------------------------------- staged breakers
def test_staged_circuit_breakers():
    """Simulate -1%, -2%, -3%, -5% drawdowns; assert the correct staged action.

    Each level is a DISTINCT action (not binary close-all):
      -1% -> reduce new sizes to 0.75x (new positions still allowed)
      -2% -> halve existing positions, block new
      -3% -> close the single weakest-correlated position only
      -5% -> close all positions, halt the session
    """
    nav, entry, stop = 100_000.0, 100.0, 98.0

    # Level 1: -1% -> 0.75x new-size multiplier, entries still allowed.
    rm = RiskManager()
    intended = rm.calculate_position_size(nav, entry, stop)
    r1 = rm.update(intraday_drawdown=-0.01)
    assert r1.level == 1
    assert rm.allow_new_position() is True
    assert rm.calculate_position_size(nav, entry, stop) == pytest.approx(intended * 0.75)

    # Level 2: -2% -> halve existing, block new.
    rm = RiskManager()
    rm.positions = {"A": {"notional": 10_000.0}}
    r2 = rm.update(intraday_drawdown=-0.02)
    assert r2.level == 2
    assert rm.positions["A"]["notional"] == 5_000.0
    assert rm.allow_new_position() is False

    # Level 3: -3% -> close ONLY the single weakest-correlated position.
    rm = RiskManager()
    rm.positions = make_positions()  # A weakly correlated; B,C tightly correlated
    corr = pd.DataFrame(
        [[1.0, 0.1, 0.1], [0.1, 1.0, 0.9], [0.1, 0.9, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    r3 = rm.update(intraday_drawdown=-0.03, regime="neutral", correlations=corr)
    assert r3.level == 3
    assert r3.closed_positions == ["A"]
    assert len(rm.positions) == 2  # the other two are NOT closed at this level

    # Level 4: -5% -> close all, halt the session.
    rm = RiskManager()
    rm.positions = make_positions()
    r4 = rm.update(intraday_drawdown=-0.05)
    assert r4.level == 4
    assert rm.positions == {}
    assert r4.halted is True
    assert set(r4.closed_positions) == {"A", "B", "C"}


def test_lockfile_created(isolated_lockfile):
    """A -10% drawdown from peak writes the lockfile and locks trading."""
    rm = RiskManager()
    rm.positions = make_positions()
    result = rm.update(intraday_drawdown=-0.10, drawdown_from_peak=-0.10)

    assert isolated_lockfile.exists()
    assert result.locked is True
    assert rm.positions == {}


def test_lockfile_blocks_trading(isolated_lockfile):
    """While the lockfile exists, order_executor refuses ALL orders."""
    isolated_lockfile.write_text("locked")
    # Regardless of drawdown / side / size, submission is refused.
    with pytest.raises(oe.OrderRejectedError):
        oe.submit_order("SPY", 10, "buy")
    with pytest.raises(oe.OrderRejectedError):
        oe.submit_order("QQQ", 999, "sell")


# ----------------------------------------------------------------- gap open
def test_check_gap_risk():
    """Gap risk fires above the threshold, not below; threshold is tunable."""
    assert check_gap_risk(open_price=103.0, prev_close=100.0) is True   # +3% > 2%
    assert check_gap_risk(open_price=101.0, prev_close=100.0) is False  # +1% < 2%
    assert check_gap_risk(open_price=97.0, prev_close=100.0) is True    # -3% (abs)
    # Custom threshold.
    assert check_gap_risk(101.0, 100.0, threshold=0.005) is True


def test_handle_gap_open_halves_and_blocks():
    """A gap-open response halves existing positions and blocks new entries."""
    rm = RiskManager()
    rm.positions = make_positions()  # three 10k positions
    result = rm.handle_gap_open()

    assert result["positions_reduced"] == 3
    assert all(p["notional"] == 5_000.0 for p in rm.positions.values())
    assert rm.allow_new_position() is False


# ----------------------------------------------------- time-window breakers
def test_weekly_drawdown_caps_leverage_for_5_days():
    """A -3% weekly loss caps leverage at 1.0x for 5 trading days, then restores."""
    rm = RiskManager()
    assert rm.effective_max_leverage() == 1.25  # MAX_PORTFOLIO_LEVERAGE

    result = rm.check_drawdown_limits(weekly_drawdown=-0.03)
    assert result.weekly_breach is True
    assert rm.effective_max_leverage() == 1.0
    assert rm.leverage_override_days_left == 5

    # Sizing now respects the reduced cap: no exposure beyond 1.0x NAV.
    size = rm.calculate_position_size(100_000.0, 100.0, 99.0)
    assert size <= 1.0 * 100_000.0 + 1e-6

    # Counts down over 5 days, then the normal cap returns.
    for _ in range(5):
        rm.tick_day()
    assert rm.leverage_override_days_left == 0
    assert rm.effective_max_leverage() == 1.25


def test_weekly_drawdown_no_breach_above_threshold():
    rm = RiskManager()
    result = rm.check_drawdown_limits(weekly_drawdown=-0.02)  # shallower than -3%
    assert result.weekly_breach is False
    assert rm.effective_max_leverage() == 1.25


def test_monthly_drawdown_halves_positions_and_alerts():
    """A -7% rolling-month loss halves positions and queues a webhook alert."""
    rm = RiskManager()
    rm.positions = make_positions()  # three 10k positions
    result = rm.check_drawdown_limits(monthly_drawdown=-0.07)

    assert result.monthly_breach is True
    assert result.positions_reduced == 3
    assert all(p["notional"] == 5_000.0 for p in rm.positions.values())
    assert result.alert is not None
    assert result.alert["alert_type"] == "circuit_breaker"
    assert result.alert["data"]["reason"] == "rolling_month_drawdown"


def test_monthly_drawdown_no_breach():
    rm = RiskManager()
    rm.positions = make_positions()
    result = rm.check_drawdown_limits(monthly_drawdown=-0.05)  # shallower than -7%
    assert result.monthly_breach is False
    assert result.alert is None
    assert all(p["notional"] == 10_000.0 for p in rm.positions.values())


# ----------------------------------------------------------------- gap entry
def test_overnight_gap_blocks_entry():
    """A 3% overnight gap blocks new entries for the session.

    Both the live gap-open response (block new entries) and the predicate that
    drives it are exercised.
    """
    assert check_gap_risk(open_price=103.0, prev_close=100.0) is True  # +3% gap

    rm = RiskManager()
    rm.positions = make_positions()
    rm.handle_gap_open()  # session response to the detected gap

    # No new entries this session, and sizing returns 0 for any candidate.
    assert rm.allow_new_position() is False
    assert rm.calculate_position_size(100_000.0, 100.0, 98.0) == 0.0


# --------------------------------------------------------------- correlation
def test_stress_correlation():
    """Effective correlation exceeds the raw correlation in a crash regime."""
    raw = 0.5
    effective_crash = get_effective_correlation(raw, "crash")
    assert effective_crash > raw  # correlations spike upward under stress
    # ...and crash stress exceeds bull (where the multiplier is < 1).
    assert effective_crash > get_effective_correlation(raw, "bull")


def test_correlated_group_identifies_high_corr_positions():
    """Existing positions with effective corr > 0.70 form the candidate's group."""
    rm = RiskManager()
    rm.positions = {
        "QQQ": {"notional": 10_000.0, "sector": "tech"},
        "GLD": {"notional": 10_000.0, "sector": "commodity"},
    }
    # Raw corr: SPY-QQQ 0.85 (high), SPY-GLD 0.10 (low). Neutral regime (1.0x).
    corr = pd.DataFrame(
        [[1.0, 0.85, 0.10], [0.85, 1.0, 0.20], [0.10, 0.20, 1.0]],
        index=["SPY", "QQQ", "GLD"],
        columns=["SPY", "QQQ", "GLD"],
    )
    group = rm.correlated_group("SPY", "neutral", corr)
    assert group == ["QQQ"]  # GLD not correlated enough


def test_correlation_grouping_caps_size():
    """A new position correlated > 0.70 with an existing one shares the cap.

    QQQ already holds 14% of NAV; SPY is 0.85-correlated, so they share the 15%
    per-position cap, leaving only ~1% of NAV for SPY.
    """
    nav = 100_000.0
    corr = pd.DataFrame(
        [[1.0, 0.85], [0.85, 1.0]], index=["SPY", "QQQ"], columns=["SPY", "QQQ"]
    )
    rm = RiskManager()
    rm.positions = {"QQQ": {"notional": 14_000.0, "sector": "tech"}}

    size = rm.calculate_position_size(
        nav, 100.0, 99.0, symbol="SPY", regime="neutral", correlations=corr
    )
    # Group cap: 15% of NAV (15k) - 14k already used = 1k remaining.
    assert size == pytest.approx(1_000.0)


def test_correlation_grouping_no_cap_when_uncorrelated():
    """An uncorrelated candidate is not constrained by the group cap."""
    nav = 100_000.0
    corr = pd.DataFrame(
        [[1.0, 0.10], [0.10, 1.0]], index=["SPY", "GLD"], columns=["SPY", "GLD"]
    )
    rm = RiskManager()
    rm.positions = {"GLD": {"notional": 14_000.0, "sector": "commodity"}}
    size = rm.calculate_position_size(
        nav, 100.0, 99.0, symbol="SPY", regime="neutral", correlations=corr
    )
    # No grouping; capped only by the per-position 15% NAV cap.
    assert size == pytest.approx(MAX_POSITION_SIZE_PCT * nav)


# -------------------------------------------------------------- sizing cap
def test_max_position_size():
    """No single position exceeds MAX_POSITION_SIZE_PCT (15%) of NAV."""
    rm = RiskManager()
    rng = np.random.default_rng(0)
    for _ in range(300):
        nav = float(rng.uniform(10_000, 1_000_000))
        entry = float(rng.uniform(10, 500))
        stop = entry * (1 - rng.uniform(0.005, 0.10))  # stop below entry
        size = rm.calculate_position_size(nav, entry, stop)
        assert size <= MAX_POSITION_SIZE_PCT * nav + 1e-6
