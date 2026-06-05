"""Order execution and lifecycle management.

Submits and manages Alpaca orders through an injected ``TradingClient`` (so the
executor is unit-testable with a mock). Every order-submitting call first checks
the circuit-breaker lockfile; while it exists, submission is refused.

Two refusal styles coexist intentionally:

* The legacy module-level :func:`submit_order` / :func:`pretrade_check` *raise*
  ``OrderRejectedError`` -- this is the hard safety gate relied on elsewhere.
* The :class:`OrderExecutor` submit methods *return None* on a lockfile (and log
  a warning), so the trading loop degrades gracefully rather than crashing.

Partial fills are first-class: ``get_order_status`` reports a
``partially_filled`` order explicitly with separate filled/remaining quantities
and never treats it as a failure.
"""

import logging

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopOrderRequest,
)

from broker.alpaca_client import with_retries
from core.risk_manager import LOCKFILE_PATH, check_lockfile
from monitoring.logging_config import get_logger

logger = get_logger("order_executor")

# stdlib logger kept for the legacy gate (used by existing risk-manager tests).
_stdlib_logger = logging.getLogger(__name__)


# --------------------------------------------------------------- legacy gate
class OrderRejectedError(RuntimeError):
    """Raised when an order is refused by a pre-trade safety check."""


def pretrade_check() -> None:
    """Raise ``OrderRejectedError`` if trading is locked out by the lockfile."""
    if check_lockfile():
        raise OrderRejectedError(
            f"Trading halted: lockfile present at {LOCKFILE_PATH}. "
            "Delete it manually to resume."
        )


def submit_order(symbol: str, qty: float, side: str, *, order_type: str = "market"):
    """Legacy pre-trade-gated submit. Raises ``OrderRejectedError`` if locked."""
    pretrade_check()
    _stdlib_logger.info("Pre-trade check passed for %s %s %s (%s)", side, qty, symbol, order_type)
    raise NotImplementedError(
        "Use OrderExecutor for real submission; this legacy entry point only gates."
    )


# --------------------------------------------------------------- executor
class OrderExecutor:
    """Submit and manage orders via an injected Alpaca ``TradingClient``."""

    def __init__(self, trading_client) -> None:
        self._client = trading_client

    # -- internal helpers ----------------------------------------------------
    def _blocked_by_lockfile(self, operation: str, **context) -> bool:
        if check_lockfile():
            logger.warning("order_blocked_lockfile", operation=operation, **context)
            return True
        return False

    def _submit(self, request, operation: str, **context):
        """Submit a prepared order request; return order_id str or None."""
        if self._blocked_by_lockfile(operation, **context):
            return None
        try:
            order = with_retries(
                self._client.submit_order, order_data=request, operation=operation
            )
        except Exception as exc:
            logger.error("order_submit_failed", operation=operation, error=str(exc), **context)
            return None
        order_id = str(order.id)
        logger.info("order_submitted", operation=operation, order_id=order_id, **context)
        return order_id

    # -- public API ----------------------------------------------------------
    def submit_market_order(self, symbol: str, qty: int, side: OrderSide):
        """Submit a market order; return the order id (str) or None if blocked."""
        request = MarketOrderRequest(
            symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY
        )
        return self._submit(request, "submit_market_order", symbol=symbol, qty=qty, side=str(side))

    def submit_limit_order(self, symbol: str, qty: int, side: OrderSide, limit_price: float):
        """Submit a limit order; return the order id (str) or None if blocked."""
        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
        )
        return self._submit(
            request, "submit_limit_order", symbol=symbol, qty=qty,
            side=str(side), limit_price=limit_price,
        )

    def set_stop_loss(self, symbol: str, stop_price: float):
        """Place a stop order to exit the current position in ``symbol``."""
        if self._blocked_by_lockfile("set_stop_loss", symbol=symbol):
            return None
        try:
            position = with_retries(
                self._client.get_open_position, symbol, operation="get_open_position"
            )
        except Exception as exc:
            logger.error("stop_loss_no_position", symbol=symbol, error=str(exc))
            return None

        qty = abs(int(float(position.qty)))
        if qty == 0:
            logger.warning("stop_loss_zero_qty", symbol=symbol)
            return None
        # Exit side is opposite the position's direction.
        side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY
        request = StopOrderRequest(
            symbol=symbol, qty=qty, side=side,
            stop_price=stop_price, time_in_force=TimeInForce.DAY,
        )
        return self._submit(request, "set_stop_loss", symbol=symbol, stop_price=stop_price)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by id; return True on success, False otherwise."""
        try:
            with_retries(
                self._client.cancel_order_by_id, order_id, operation="cancel_order"
            )
        except Exception as exc:
            logger.error("order_cancel_failed", order_id=order_id, error=str(exc))
            return False
        logger.info("order_cancelled", order_id=order_id)
        return True

    def get_order_status(self, order_id: str) -> dict:
        """Return ``{status, filled_qty, remaining_qty}`` for an order.

        Handles ``partially_filled`` explicitly: it is a normal state, not a
        failure, and the filled/remaining quantities are logged separately.
        """
        order = with_retries(
            self._client.get_order_by_id, order_id, operation="get_order_status"
        )
        status = order.status.value if hasattr(order.status, "value") else str(order.status)
        total_qty = int(float(order.qty))
        filled_qty = int(float(order.filled_qty or 0))
        remaining_qty = max(0, total_qty - filled_qty)

        result = {
            "status": status,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
        }

        if status == "partially_filled":
            logger.info(
                "order_partially_filled",
                order_id=order_id,
                status=status,
                filled_qty=filled_qty,
                remaining_qty=remaining_qty,
            )
        else:
            logger.info(
                "order_status",
                order_id=order_id,
                status=status,
                filled_qty=filled_qty,
                remaining_qty=remaining_qty,
            )
        return result
