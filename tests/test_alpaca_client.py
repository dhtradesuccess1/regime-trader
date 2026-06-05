"""Tests for ``broker.alpaca_client`` (retry helper + client wrapper)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import broker.alpaca_client as ac
from broker.alpaca_client import AlpacaClient, with_retries


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    monkeypatch.setattr(ac.time, "sleep", lambda _s: None)


def test_with_retries_success_first_try():
    fn = MagicMock(return_value="ok")
    assert with_retries(fn, operation="t") == "ok"
    assert fn.call_count == 1


def test_with_retries_succeeds_after_failures():
    fn = MagicMock(side_effect=[ValueError("x"), "ok"])
    assert with_retries(fn, operation="t") == "ok"
    assert fn.call_count == 2


def test_with_retries_exhausts_and_raises():
    fn = MagicMock(side_effect=RuntimeError("down"))
    with pytest.raises(RuntimeError):
        with_retries(fn, operation="t")
    assert fn.call_count == 3


def _make_client(monkeypatch, account=None, clock=None):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    fake_trading = MagicMock()
    fake_trading.get_account.return_value = account
    fake_trading.get_clock.return_value = clock
    monkeypatch.setattr(ac, "TradingClient", lambda *a, **k: fake_trading)
    monkeypatch.setattr(ac, "load_dotenv", lambda *a, **k: None)
    return AlpacaClient()


def test_init_raises_without_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(ac, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        AlpacaClient()


def test_get_buying_power(monkeypatch):
    client = _make_client(monkeypatch, account=SimpleNamespace(buying_power="25000"))
    assert client.get_buying_power() == 25000.0


def test_is_market_open(monkeypatch):
    client = _make_client(monkeypatch, clock=SimpleNamespace(is_open=True))
    assert client.is_market_open() is True
