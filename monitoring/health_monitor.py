"""System health monitor.

Assesses the health of the trading system from a snapshot of runtime signals
and classifies severity. By default it uses a fast, dependency-free heuristic;
if an ``ANTHROPIC_API_KEY`` is configured a richer LLM summary can be layered on
later. When severity warrants it, an alert is dispatched via
:mod:`monitoring.alerts`.
"""

import os

from dotenv import load_dotenv

from monitoring.logging_config import get_logger

logger = get_logger("health_monitor")

# Severity ordering for comparisons.
SEVERITY_ORDER = {"ok": 0, "warning": 1, "critical": 2}


def assess_health(state: dict) -> dict:
    """Return a health assessment ``{severity, findings}`` from a state snapshot.

    Recognized state keys (all optional): ``intraday_drawdown``,
    ``peak_drawdown``, ``data_valid`` (bool), ``regime_stable`` (bool),
    ``execution_errors`` (int), ``locked`` (bool).
    """
    findings: list[str] = []
    severity = "ok"

    def escalate(level: str) -> None:
        nonlocal severity
        if SEVERITY_ORDER[level] > SEVERITY_ORDER[severity]:
            severity = level

    if state.get("locked"):
        findings.append("Trading is locked out (lockfile present).")
        escalate("critical")

    intraday = state.get("intraday_drawdown")
    if intraday is not None and intraday <= -0.03:
        findings.append(f"Large intraday drawdown: {intraday:.2%}.")
        escalate("critical" if intraday <= -0.05 else "warning")

    peak = state.get("peak_drawdown")
    if peak is not None and peak <= -0.08:
        findings.append(f"Deep drawdown from peak: {peak:.2%}.")
        escalate("critical")

    if state.get("data_valid") is False:
        findings.append("Latest market data failed validation.")
        escalate("warning")

    if state.get("regime_stable") is False:
        findings.append("Regime is unstable (frequent flips).")
        escalate("warning")

    errors = state.get("execution_errors", 0)
    if errors:
        findings.append(f"{errors} execution error(s) since last check.")
        escalate("warning" if errors < 3 else "critical")

    if not findings:
        findings.append("All systems nominal.")

    assessment = {"severity": severity, "findings": findings}
    logger.info("health_assessment", severity=severity, findings=findings)
    return assessment


def anthropic_available() -> bool:
    """Whether an Anthropic API key is configured (for the optional LLM path)."""
    load_dotenv()
    key = os.getenv("ANTHROPIC_API_KEY")
    return bool(key) and key != "your_anthropic_key_here"
