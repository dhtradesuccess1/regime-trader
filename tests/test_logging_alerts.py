"""Tests for ``monitoring.logging_config`` and ``monitoring.alerts``."""

import asyncio
import json

import pytest

import monitoring.alerts as alerts_mod
from monitoring.alerts import send_alert
from monitoring.logging_config import (
    bind_trading_context,
    clear_trading_context,
    configure_logging,
    get_logger,
)


def test_logging_writes_jsonl_with_required_fields(tmp_path):
    configure_logging(log_level="INFO", log_dir=str(tmp_path))
    log = get_logger("test")
    clear_trading_context()
    log.info("plain_event")
    bind_trading_context(regime="bull", confidence=0.7)
    log.info("bound_event")

    import logging
    for h in logging.getLogger().handlers:
        h.flush()

    files = sorted(tmp_path.glob("trading_*.jsonl"))
    assert files
    records = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    required = {"ts", "event", "regime", "confidence", "portfolio_value",
                "drawdown_pct", "active_positions"}
    for rec in records:
        assert required.issubset(rec.keys())
    plain = next(r for r in records if r["event"] == "plain_event")
    assert plain["regime"] == "unknown"
    bound = next(r for r in records if r["event"] == "bound_event")
    assert bound["regime"] == "bull" and bound["confidence"] == 0.7
    clear_trading_context()


def test_error_event_has_error_fields(tmp_path):
    configure_logging(log_level="INFO", log_dir=str(tmp_path))
    log = get_logger("test")
    clear_trading_context()
    try:
        raise ValueError("boom")
    except ValueError:
        log.error("failure", exc_info=True)
    import logging
    for h in logging.getLogger().handlers:
        h.flush()
    files = sorted(tmp_path.glob("trading_*.jsonl"))
    records = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
    err = next(r for r in records if r["event"] == "failure")
    assert err["error_type"] == "ValueError"
    assert "Traceback" in err["stack_trace"]


def test_alert_skips_when_unset(monkeypatch):
    monkeypatch.setattr(alerts_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("MAKE_WEBHOOK_URL", raising=False)
    # Must complete without error and without attempting a request.
    asyncio.run(send_alert("regime_change", {"x": 1}))


def test_alert_invalid_type(monkeypatch):
    monkeypatch.setattr(alerts_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("MAKE_WEBHOOK_URL", "https://example.com/hook")
    # Invalid type is logged and skipped, never raises.
    asyncio.run(send_alert("not_valid", {}))


def test_alert_converts_newlines_to_br(monkeypatch):
    """Newlines in the payload become <br> so Make.com renders line breaks."""
    monkeypatch.setattr(alerts_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("MAKE_WEBHOOK_URL", "https://example.com/hook")

    captured = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            captured["payload"] = json

            class Resp:
                def raise_for_status(self):
                    return None

            return Resp()

    monkeypatch.setattr(alerts_mod.httpx, "AsyncClient", FakeClient)

    body = "All systems nominal.\nRegime: bull\r\nConfidence: 0.9"
    asyncio.run(
        send_alert("health_alert", {"body": body, "findings": ["line1\nline2"]})
    )

    sent = captured["payload"]
    assert "\n" not in sent["body"] and "\r" not in sent["body"]
    assert sent["body"] == "All systems nominal.<br>Regime: bull<br>Confidence: 0.9"
    # Nested list strings are converted too.
    assert sent["findings"] == ["line1<br>line2"]


def test_render_newlines_unit():
    """The helper converts strings recursively and leaves non-strings alone."""
    out = alerts_mod.render_newlines({"a": "x\ny", "b": [{"c": "p\r\nq"}], "n": 3})
    assert out == {"a": "x<br>y", "b": [{"c": "p<br>q"}], "n": 3}


def test_alert_retries_then_gives_up(monkeypatch):
    monkeypatch.setattr(alerts_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("MAKE_WEBHOOK_URL", "https://example.com/hook")

    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            calls["n"] += 1
            raise alerts_mod.httpx.ConnectError("refused")

    monkeypatch.setattr(alerts_mod.httpx, "AsyncClient", FakeClient)

    async def no_sleep(_s):
        return None

    monkeypatch.setattr(alerts_mod.asyncio, "sleep", no_sleep)

    asyncio.run(send_alert("circuit_breaker", {"level": 4}))  # never raises
    assert calls["n"] == 3
