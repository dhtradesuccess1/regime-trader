"""Live position and exposure tracking.

Reads current holdings and account equity from Alpaca (via an injected
:class:`~broker.alpaca_client.AlpacaClient`) and maintains the drawdown
references the risk manager consumes:

* intraday drawdown -- current NAV vs. the session-open NAV.
* peak drawdown -- current NAV vs. the all-time peak NAV.
* weekly / monthly drawdown -- current NAV vs. the rolling peak over the trailing
  5 / 21 trading days (feeds the risk manager's time-window breakers).

Call :meth:`PositionTracker.record_nav` once per trading day to keep the peak
and the rolling NAV history current.
"""

from collections import deque

from monitoring.logging_config import get_logger

logger = get_logger("position_tracker")

# Rolling NAV history windows (trading days).
TRADING_WEEK_DAYS = 5
TRADING_MONTH_DAYS = 21


def _position_to_dict(position) -> dict:
    """Convert an Alpaca Position object into a plain dict."""
    return {
        "symbol": position.symbol,
        "qty": float(position.qty),
        "avg_entry_price": float(position.avg_entry_price),
        "market_value": float(position.market_value),
        "current_price": float(position.current_price),
        "unrealized_pl": float(position.unrealized_pl),
        "side": getattr(position.side, "value", str(position.side)),
    }


class PositionTracker:
    """Tracks open positions and NAV-based drawdowns."""

    def __init__(self, client) -> None:
        self._client = client
        self.session_open_nav: float | None = None
        self.peak_nav: float | None = None
        # Rolling daily NAV history (newest last); a month covers the week too.
        self.nav_history: deque[float] = deque(maxlen=TRADING_MONTH_DAYS)

    # ------------------------------------------------------------- positions
    def get_open_positions(self) -> list[dict]:
        """Return all open positions as a list of dicts."""
        positions = self._client.trading_client.get_all_positions()
        return [_position_to_dict(p) for p in positions]

    def get_position(self, symbol: str) -> dict | None:
        """Return a single position dict, or None if there is no open position."""
        try:
            position = self._client.trading_client.get_open_position(symbol)
        except Exception:
            # alpaca-py raises if there is no open position for the symbol.
            return None
        return _position_to_dict(position)

    # ----------------------------------------------------------------- NAV
    def _current_nav(self) -> float:
        return float(self._client.get_account().equity)

    def start_session(self) -> None:
        """Record the session-open NAV (call at the start of each session)."""
        nav = self._current_nav()
        self.session_open_nav = nav
        self.update_peak_nav(nav)
        logger.info("session_started", session_open_nav=nav, peak_nav=self.peak_nav)

    def record_nav(self, nav: float | None = None) -> float:
        """Append the day's NAV to the rolling history and update the peak.

        Call once per trading day. If ``nav`` is omitted it is fetched from the
        account. Returns the recorded NAV.
        """
        if nav is None:
            nav = self._current_nav()
        self.nav_history.append(nav)
        self.update_peak_nav(nav)
        return nav

    def calculate_intraday_drawdown(self) -> float:
        """Signed drawdown of current NAV vs. session-open NAV (<= 0 when down)."""
        nav = self._current_nav()
        if self.session_open_nav is None:
            self.session_open_nav = nav
        if self.session_open_nav <= 0:
            return 0.0
        return nav / self.session_open_nav - 1.0

    def calculate_peak_drawdown(self) -> float:
        """Signed drawdown of current NAV vs. all-time peak NAV (<= 0 when down)."""
        nav = self._current_nav()
        if self.peak_nav is None:
            self.peak_nav = nav
        if self.peak_nav <= 0:
            return 0.0
        return nav / self.peak_nav - 1.0

    def update_peak_nav(self, current_nav: float) -> None:
        """Raise the recorded peak NAV if the current NAV is a new high."""
        if self.peak_nav is None or current_nav > self.peak_nav:
            self.peak_nav = current_nav

    def _rolling_drawdown(self, window_days: int) -> float:
        """Signed drawdown of the latest NAV vs. the rolling peak over a window."""
        if not self.nav_history:
            return 0.0
        window = list(self.nav_history)[-window_days:]
        current = window[-1]
        peak = max(window)
        if peak <= 0:
            return 0.0
        return current / peak - 1.0

    def calculate_weekly_drawdown(self) -> float:
        """Signed drawdown vs. the rolling peak over the trailing 5 days (<= 0)."""
        return self._rolling_drawdown(TRADING_WEEK_DAYS)

    def calculate_monthly_drawdown(self) -> float:
        """Signed drawdown vs. the rolling peak over the trailing 21 days (<= 0)."""
        return self._rolling_drawdown(TRADING_MONTH_DAYS)
