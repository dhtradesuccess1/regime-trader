"""Tests for ``broker.order_executor``.

Covers the lockfile submission gate, explicit partial-fill handling and
logging, API retry behavior, and the happy-path order-id return. The Alpaca
``TradingClient`` is replaced with a mock so no network or credentials are
needed.
"""

from unittest.mock import MagicMock

import pytest
import structlog
from alpaca.trading.enums import OrderSide, OrderStatus

import broker.alpaca_client as alpaca_client
import core.risk_manager as rm_mod
from broker.order_executor import OrderExecutor


@pytest.fixture(autouse=True)
def isolated_lockfile(tmp_path, monkeypatch):
    """Redirect the lockfile to a temp path for every test."""
    lock = tmp_path / "lockfile.lock"
    monkeypatch.setattr(rm_mod, "LOCKFILE_PATH", lock)
    return lock


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    """Make retry backoff instant so the retry test runs fast."""
    monkeypatch.setattr(alpaca_client.time, "sleep", lambda _seconds: None)


def make_order(order_id="order-123", status=OrderStatus.FILLED, qty="10", filled="10"):
    order = MagicMock()
    order.id = order_id
    order.status = status
    order.qty = qty
    order.filled_qty = filled
    return order


def test_lockfile_blocks_submission(isolated_lockfile):
    """With the lockfile present, submit returns None and never calls Alpaca."""
    isolated_lockfile.write_text("locked")
    client = MagicMock()
    executor = OrderExecutor(client)

    result = executor.submit_market_order("SPY", 10, OrderSide.BUY)

    assert result is None
    client.submit_order.assert_not_called()


def test_partial_fill_logged():
    """A partially_filled order logs both filled_qty and remaining_qty."""
    client = MagicMock()
    client.get_order_by_id.return_value = make_order(
        status=OrderStatus.PARTIALLY_FILLED, qty="10", filled="3"
    )
    executor = OrderExecutor(client)

    with structlog.testing.capture_logs() as logs:
        result = executor.get_order_status("order-123")

    # Status reported explicitly, not as a failure.
    assert result == {"status": "partially_filled", "filled_qty": 3, "remaining_qty": 7}

    partial_logs = [e for e in logs if e.get("event") == "order_partially_filled"]
    assert partial_logs, f"no partial-fill log emitted; got {logs}"
    entry = partial_logs[0]
    assert entry["filled_qty"] == 3
    assert entry["remaining_qty"] == 7


def test_retry_on_api_error():
    """A persistently failing submit is retried exactly 3 times, then None."""
    client = MagicMock()
    client.submit_order.side_effect = ConnectionError("api down")
    executor = OrderExecutor(client)

    result = executor.submit_market_order("SPY", 5, OrderSide.BUY)

    assert result is None
    assert client.submit_order.call_count == 3


def test_market_order_returns_order_id():
    """A successful market order returns the order id as a string."""
    client = MagicMock()
    client.submit_order.return_value = make_order(order_id="abc-987")
    executor = OrderExecutor(client)

    result = executor.submit_market_order("QQQ", 2, OrderSide.BUY)

    assert isinstance(result, str)
    assert result == "abc-987"
    client.submit_order.assert_called_once()


def test_limit_order_returns_order_id():
    client = MagicMock()
    client.submit_order.return_value = make_order(order_id="lim-1")
    executor = OrderExecutor(client)
    assert executor.submit_limit_order("SPY", 3, OrderSide.BUY, 400.0) == "lim-1"


def test_full_fill_status_not_partial():
    """A fully filled order reports remaining_qty 0 and no partial log."""
    client = MagicMock()
    client.get_order_by_id.return_value = make_order(
        status=OrderStatus.FILLED, qty="10", filled="10"
    )
    executor = OrderExecutor(client)
    with structlog.testing.capture_logs() as logs:
        result = executor.get_order_status("o-1")
    assert result == {"status": "filled", "filled_qty": 10, "remaining_qty": 0}
    assert not [e for e in logs if e.get("event") == "order_partially_filled"]


def test_cancel_order_success_and_failure():
    client = MagicMock()
    executor = OrderExecutor(client)
    assert executor.cancel_order("o-1") is True

    client.cancel_order_by_id.side_effect = RuntimeError("nope")
    assert executor.cancel_order("o-2") is False


def test_set_stop_loss(monkeypatch):
    client = MagicMock()
    client.get_open_position.return_value = MagicMock(qty="10")
    client.submit_order.return_value = make_order(order_id="stop-1")
    executor = OrderExecutor(client)
    assert executor.set_stop_loss("SPY", 380.0) == "stop-1"


def test_set_stop_loss_no_position():
    client = MagicMock()
    client.get_open_position.side_effect = Exception("no position")
    executor = OrderExecutor(client)
    assert executor.set_stop_loss("SPY", 380.0) is None


def test_stop_loss_blocked_by_lockfile(isolated_lockfile):
    isolated_lockfile.write_text("locked")
    client = MagicMock()
    executor = OrderExecutor(client)
    assert executor.set_stop_loss("SPY", 380.0) is None
    client.submit_order.assert_not_called()
