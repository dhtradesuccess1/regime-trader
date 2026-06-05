"""Tests for ``broker.position_tracker`` using a mocked Alpaca client."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from broker.position_tracker import PositionTracker


def make_position(symbol="SPY", qty="10"):
    return SimpleNamespace(
        symbol=symbol,
        qty=qty,
        avg_entry_price="400",
        market_value="4000",
        current_price="405",
        unrealized_pl="50",
        side=SimpleNamespace(value="long"),
    )


def make_client(equity=100_000.0, positions=None):
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity=str(equity))
    client.trading_client.get_all_positions.return_value = positions or []
    return client


def test_get_open_positions():
    client = make_client(positions=[make_position("SPY"), make_position("QQQ")])
    tracker = PositionTracker(client)
    positions = tracker.get_open_positions()
    assert {p["symbol"] for p in positions} == {"SPY", "QQQ"}
    assert positions[0]["qty"] == 10.0


def test_get_position_found_and_missing():
    client = make_client()
    client.trading_client.get_open_position.return_value = make_position("SPY")
    tracker = PositionTracker(client)
    assert tracker.get_position("SPY")["symbol"] == "SPY"

    client.trading_client.get_open_position.side_effect = Exception("no position")
    assert tracker.get_position("NONE") is None


def test_intraday_drawdown():
    client = make_client(equity=98_000.0)
    tracker = PositionTracker(client)
    tracker.session_open_nav = 100_000.0
    assert tracker.calculate_intraday_drawdown() == pytest.approx(-0.02)


def test_peak_drawdown_and_update():
    client = make_client(equity=90_000.0)
    tracker = PositionTracker(client)
    tracker.update_peak_nav(100_000.0)
    tracker.update_peak_nav(95_000.0)  # not a new high
    assert tracker.peak_nav == 100_000.0
    assert tracker.calculate_peak_drawdown() == pytest.approx(-0.10)


def test_start_session_sets_references():
    client = make_client(equity=100_000.0)
    tracker = PositionTracker(client)
    tracker.start_session()
    assert tracker.session_open_nav == 100_000.0
    assert tracker.peak_nav == 100_000.0


def test_record_nav_updates_history_and_peak():
    tracker = PositionTracker(make_client())
    tracker.record_nav(100_000.0)
    tracker.record_nav(102_000.0)
    tracker.record_nav(101_000.0)
    assert list(tracker.nav_history) == [100_000.0, 102_000.0, 101_000.0]
    assert tracker.peak_nav == 102_000.0


def test_weekly_and_monthly_drawdown():
    tracker = PositionTracker(make_client())
    # Exactly 21 days (fills the deque). Monthly peak (110k) is early and sits
    # OUTSIDE the trailing-5 weekly window, whose peak is 108k.
    navs = (
        [110_000] + [104_000] * 15 + [108_000, 106_000, 105_000, 103_000, 102_000]
    )
    assert len(navs) == 21
    for nav in navs:
        tracker.record_nav(float(nav))

    # Weekly window = last 5 NAVs: peak 108k, current 102k -> ~-5.6%.
    weekly = tracker.calculate_weekly_drawdown()
    assert weekly == pytest.approx(102_000 / 108_000 - 1.0)
    assert weekly < -0.03  # breaches the -3% weekly limit

    # Monthly window = all 21 NAVs: peak 110k, current 102k -> ~-7.3%.
    monthly = tracker.calculate_monthly_drawdown()
    assert monthly == pytest.approx(102_000 / 110_000 - 1.0)
    assert monthly < -0.07  # breaches the -7% monthly limit


def test_drawdowns_empty_history():
    tracker = PositionTracker(make_client())
    assert tracker.calculate_weekly_drawdown() == 0.0
    assert tracker.calculate_monthly_drawdown() == 0.0
