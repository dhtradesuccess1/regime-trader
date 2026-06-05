"""Alpaca API client wrapper.

Constructs an ``alpaca-py`` ``TradingClient`` from environment credentials
(loaded via ``python-dotenv`` -- no keys are hardcoded anywhere) and exposes a
small, defensively-wrapped surface for the rest of the system. Every API call
is logged via structlog and retried with exponential backoff (max 3 attempts).
"""

import os
import time

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from monitoring.logging_config import get_logger

logger = get_logger("alpaca_client")

# Retry policy for all outbound API calls.
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0


def with_retries(func, *args, operation: str = "api_call", max_retries: int = MAX_RETRIES, **kwargs):
    """Call ``func`` with exponential backoff; log each attempt and response.

    Retries up to ``max_retries`` times. Re-raises the last exception if all
    attempts fail (callers decide how to handle exhaustion).
    """
    delay = INITIAL_BACKOFF_SECONDS
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("api_call", operation=operation, attempt=attempt)
            result = func(*args, **kwargs)
            logger.info("api_response", operation=operation, attempt=attempt, ok=True)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "api_call_failed",
                operation=operation,
                attempt=attempt,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    logger.error("api_call_exhausted", operation=operation, error=str(last_exc))
    raise last_exc


def _paper_from_env() -> bool:
    return os.getenv("ALPACA_PAPER", "true").strip().lower() in ("true", "1", "yes")


class AlpacaClient:
    """Thin, retrying wrapper around ``alpaca-py``'s ``TradingClient``."""

    def __init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set in environment (.env)."
            )
        paper = _paper_from_env()
        logger.info("alpaca_client_init", paper=paper)
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

    def get_account(self):
        """Return the Alpaca ``Account`` object."""
        return with_retries(self.trading_client.get_account, operation="get_account")

    def get_buying_power(self) -> float:
        """Return available buying power as a float."""
        account = self.get_account()
        return float(account.buying_power)

    def is_market_open(self) -> bool:
        """Return whether the market is currently open."""
        clock = with_retries(self.trading_client.get_clock, operation="get_clock")
        return bool(clock.is_open)
