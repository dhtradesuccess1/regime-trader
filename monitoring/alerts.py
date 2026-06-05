"""Alert dispatch to an external webhook (Make.com) over HTTP.

Posts JSON alerts to ``MAKE_WEBHOOK_URL`` (from ``.env``) with retries and
exponential backoff. Failures are logged via structlog and never raised, so a
flaky webhook can never crash the trading loop. If the webhook is not
configured, alerts are silently skipped.

Newline handling
----------------
Make.com's email module renders the body as HTML, so literal ``\\n`` characters
show up verbatim instead of as line breaks. Before sending, every string in the
payload has its newlines converted to ``<br>`` so multi-line bodies render
correctly. (Alternatively, set the Make.com email module's ``bodyType`` to
``rawHtml`` -- but converting here means it works regardless of how the scenario
is configured.)
"""

import asyncio
import os
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from monitoring.logging_config import get_logger

logger = get_logger("alerts")

# Webhook placeholder from .env.example -- treated as "not configured".
_PLACEHOLDER_WEBHOOK = "your_make_webhook_url_here"

# Allowed alert types.
VALID_ALERT_TYPES = {
    "circuit_breaker",
    "regime_change",
    "lockfile_created",
    "health_alert",
    "trade_executed",
    "gap_open_detected",
}

MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
REQUEST_TIMEOUT_SECONDS = 10.0

# HTML line break Make.com's email module renders correctly.
HTML_LINE_BREAK = "<br>"


def render_newlines(obj):
    """Recursively convert newlines in any string to ``<br>`` for HTML email.

    Walks dicts and lists so nested body fields are handled too. Non-string
    values pass through unchanged.
    """
    if isinstance(obj, str):
        return obj.replace("\r\n", "\n").replace("\r", "\n").replace("\n", HTML_LINE_BREAK)
    if isinstance(obj, dict):
        return {key: render_newlines(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [render_newlines(value) for value in obj]
    return obj


async def send_alert(alert_type: str, data: dict) -> None:
    """POST an alert to the configured webhook; never raises.

    Payload is ``{"alert_type": alert_type, "timestamp": <ISO8601 UTC>, **data}``.
    Retries up to 3 times with exponential backoff. Skips silently if the
    webhook is unset, and logs (does not raise) on invalid types or failures.
    """
    load_dotenv()
    webhook_url = os.getenv("MAKE_WEBHOOK_URL")
    if not webhook_url or webhook_url == _PLACEHOLDER_WEBHOOK:
        return  # silently skip when not configured

    if alert_type not in VALID_ALERT_TYPES:
        logger.warning(
            "invalid_alert_type", alert_type=alert_type, valid=sorted(VALID_ALERT_TYPES)
        )
        return

    payload = {
        "alert_type": alert_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    # Convert newlines to <br> so Make.com's email module renders line breaks.
    payload = render_newlines(payload)

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(webhook_url, json=payload)
                response.raise_for_status()
            logger.info("alert_sent", alert_type=alert_type, attempt=attempt)
            return
        except Exception as exc:  # never let an alert failure propagate
            logger.warning(
                "alert_send_failed",
                alert_type=alert_type,
                attempt=attempt,
                error=str(exc),
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff *= 2

    logger.error("alert_send_giving_up", alert_type=alert_type)
