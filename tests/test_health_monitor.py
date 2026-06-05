"""Tests for ``monitoring.health_monitor``."""

from monitoring.health_monitor import anthropic_available, assess_health


def test_nominal_state_ok():
    result = assess_health({"data_valid": True, "regime_stable": True})
    assert result["severity"] == "ok"
    assert result["findings"] == ["All systems nominal."]


def test_locked_is_critical():
    result = assess_health({"locked": True})
    assert result["severity"] == "critical"


def test_large_drawdown_escalates():
    warn = assess_health({"intraday_drawdown": -0.04})
    assert warn["severity"] == "warning"
    crit = assess_health({"intraday_drawdown": -0.06})
    assert crit["severity"] == "critical"


def test_data_and_regime_warnings():
    result = assess_health({"data_valid": False, "regime_stable": False})
    assert result["severity"] == "warning"
    assert len(result["findings"]) >= 2


def test_execution_errors_escalate():
    assert assess_health({"execution_errors": 1})["severity"] == "warning"
    assert assess_health({"execution_errors": 5})["severity"] == "critical"


def test_anthropic_available(monkeypatch):
    import monitoring.health_monitor as hm
    monkeypatch.setattr(hm, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert anthropic_available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    assert anthropic_available() is True
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your_anthropic_key_here")
    assert anthropic_available() is False
