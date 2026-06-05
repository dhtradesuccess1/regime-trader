"""Risk management: position sizing, leverage limits, and staged circuit breakers.

This module enforces the system's risk policy. Two distinct mechanisms:

**Position sizing** (:meth:`RiskManager.calculate_position_size`) applies four
constraints *simultaneously*: per-position NAV cap, fixed fractional risk per
trade (stop distance sets share count), portfolio leverage cap, and a sector
concentration cap.

**Staged circuit breakers** (:meth:`RiskManager.update`) are *not* binary. Each
drawdown level has a distinct action, escalating from a soft size reduction up
to a permanent lockfile halt:

======  =====================================  ==========================================
Level   Trigger                                Action
======  =====================================  ==========================================
1       intraday <= CIRCUIT_BREAKER_WARN        log warning; 0.75x new-position sizes
2       intraday <= CIRCUIT_BREAKER_HALF        halve existing positions; block new
3       intraday <= CIRCUIT_BREAKER_CLOSE_WEAKEST  close single weakest-correlated; recheck
4       intraday <= CIRCUIT_BREAKER_CLOSE_ALL   close all; halt for the session
5       peak     <= CIRCUIT_BREAKER_LOCKFILE     close all; write lockfile; halt permanently
======  =====================================  ==========================================

The level-3 "recheck" happens on the *next* :meth:`update` call: a single call
closes exactly one position, so a persistent level-3 drawdown unwinds the book
one position per evaluation rather than dumping everything at once.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from settings.config import (
    CIRCUIT_BREAKER_CLOSE_ALL,
    CIRCUIT_BREAKER_CLOSE_WEAKEST,
    CIRCUIT_BREAKER_HALF,
    CIRCUIT_BREAKER_LOCKFILE,
    CIRCUIT_BREAKER_WARN,
    GAP_OPEN_THRESHOLD,
    MAX_PORTFOLIO_LEVERAGE,
    MAX_POSITION_SIZE_PCT,
    RISK_PER_TRADE_PCT,
)

logger = logging.getLogger(__name__)

# Project root and the lockfile written at circuit-breaker level 5.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE_PATH = PROJECT_ROOT / "lockfile.lock"

# Sector concentration cap (fraction of NAV). Not in config.py by design --
# it is a risk-policy constant local to the risk manager.
SECTOR_CONCENTRATION_MAX = 0.40

# Pairs whose effective correlation exceeds this are treated as the SAME
# position for the per-position concentration cap (they offer no diversification
# in the prevailing regime).
CORRELATION_GROUP_THRESHOLD = 0.70

# Level-1 size reduction multiplier (reduce new sizes by 25%).
LEVEL1_SIZE_MULTIPLIER = 0.75

# --- Time-window drawdown limits (separate from intraday/peak levels) -------
# A weekly loss this deep caps leverage at 1.0x for a cooldown period.
WEEKLY_DRAWDOWN_LIMIT = -0.03
LEVERAGE_REDUCTION_DAYS = 5
REDUCED_LEVERAGE = 1.0
# A rolling-month loss this deep alerts and halves positions.
MONTHLY_DRAWDOWN_LIMIT = -0.07
MONTHLY_SIZE_REDUCTION = 0.50

# Regime stress multipliers for effective correlation. Correlations spike in
# crashes; the extra labels (capitulation/mania) extend the 5-regime spec.
REGIME_CORR_MULTIPLIER = {
    "capitulation": 1.8,
    "crash": 1.8,
    "bear": 1.4,
    "neutral": 1.0,
    "bull": 0.9,
    "euphoria": 1.1,
    "mania": 1.1,
}


def get_effective_correlation(pair_corr: float, regime: str) -> float:
    """Stress-adjusted correlation; correlations spike in crashes.

    Multiplies the raw pair correlation by a regime multiplier and clamps the
    result to ``[-1.0, 1.0]``.
    """
    multiplier = REGIME_CORR_MULTIPLIER.get(regime, 1.0)
    return float(np.clip(pair_corr * multiplier, -1.0, 1.0))


# Fraction by which existing exposure is cut when a gap-open session is detected.
GAP_OPEN_SIZE_REDUCTION = 0.50


def check_gap_risk(
    open_price: float, prev_close: float, threshold: float = GAP_OPEN_THRESHOLD
) -> bool:
    """True if the opening gap is too large to trade normally this session.

    ``threshold`` defaults to ``GAP_OPEN_THRESHOLD`` (2%).
    """
    if prev_close == 0:
        return False
    gap_pct = abs(open_price - prev_close) / prev_close
    return gap_pct > threshold


def check_lockfile() -> bool:
    """True if ``lockfile.lock`` exists in the project root.

    Called before every order. Reads the module-level ``LOCKFILE_PATH`` at call
    time so it honors test overrides.
    """
    return LOCKFILE_PATH.exists()


@dataclass
class CircuitBreakerResult:
    """Outcome of a single circuit-breaker evaluation."""

    level: int
    action: str
    message: str = ""
    size_multiplier: float = 1.0
    block_new: bool = False
    halted: bool = False
    locked: bool = False
    closed_positions: list = field(default_factory=list)


@dataclass
class DrawdownLimitResult:
    """Outcome of the weekly/monthly time-window drawdown checks."""

    weekly_breach: bool = False
    monthly_breach: bool = False
    leverage_cap: float | None = None
    leverage_days_left: int = 0
    positions_reduced: int = 0
    # Webhook/email alert payload for the caller to dispatch, or None.
    alert: dict | None = None


class RiskManager:
    """Stateful risk controller for sizing and circuit breakers.

    Positions are held as ``{symbol: {"notional": float, "sector": str}}``.
    """

    def __init__(self) -> None:
        self.positions: dict[str, dict] = {}
        self.size_multiplier: float = 1.0
        self.block_new: bool = False
        self.halted: bool = False  # halted for the remainder of the session
        self.locked: bool = False  # permanent halt until lockfile removed
        # Time-window leverage cooldown (set by the weekly drawdown limit).
        self.leverage_cap_override: float | None = None
        self.leverage_override_days_left: int = 0

    # --------------------------------------------------------------- session
    def reset_session(self) -> None:
        """Clear intraday breaker state at the start of a new session.

        Does not clear ``locked`` -- a lockfile halt persists across sessions
        until the file is manually deleted.
        """
        self.size_multiplier = 1.0
        self.block_new = False
        self.halted = False

    def allow_new_position(self) -> bool:
        """Whether a new position may be opened right now."""
        return not (
            self.block_new or self.halted or self.locked or check_lockfile()
        )

    def handle_gap_open(self) -> dict:
        """Respond to a large opening gap: block new entries and halve existing.

        Mirrors the daily-loop gap policy -- do not enter new positions this
        session, and reduce existing position sizes by 50%. Returns a summary
        for logging/alerting (the ``gap_open_detected`` event is emitted by the
        caller, which has the gap percentage in hand).
        """
        for position in self.positions.values():
            position["notional"] *= GAP_OPEN_SIZE_REDUCTION
        self.block_new = True
        logger.warning(
            "gap_open_response: blocked new entries and reduced %d positions by %.0f%%",
            len(self.positions),
            (1 - GAP_OPEN_SIZE_REDUCTION) * 100,
        )
        return {
            "action": "gap_open",
            "block_new": True,
            "positions_reduced": len(self.positions),
            "size_reduction": GAP_OPEN_SIZE_REDUCTION,
        }

    # -------------------------------------------------- time-window breakers
    def effective_max_leverage(self) -> float:
        """Current leverage cap, honoring an active weekly-drawdown cooldown."""
        if self.leverage_cap_override is not None and self.leverage_override_days_left > 0:
            return self.leverage_cap_override
        return MAX_PORTFOLIO_LEVERAGE

    def tick_day(self) -> None:
        """Advance the leverage-cooldown countdown by one trading day.

        Call once per day. When the cooldown elapses, the normal leverage cap is
        restored.
        """
        if self.leverage_override_days_left > 0:
            self.leverage_override_days_left -= 1
            if self.leverage_override_days_left == 0:
                self.leverage_cap_override = None
                logger.info("leverage_cooldown_expired: restored normal leverage cap")

    def check_drawdown_limits(
        self, *, weekly_drawdown: float | None = None, monthly_drawdown: float | None = None
    ) -> DrawdownLimitResult:
        """Evaluate the weekly and rolling-month drawdown limits.

        * Weekly loss <= -3%: cap leverage at 1.0x for the next 5 trading days.
        * Rolling-month loss <= -7%: halve all positions and return a webhook
          alert payload for the caller to dispatch.

        Both are signed fractions (negative = loss). Returns a
        :class:`DrawdownLimitResult` describing the actions taken.
        """
        result = DrawdownLimitResult()

        if weekly_drawdown is not None and weekly_drawdown <= WEEKLY_DRAWDOWN_LIMIT:
            self.leverage_cap_override = REDUCED_LEVERAGE
            self.leverage_override_days_left = LEVERAGE_REDUCTION_DAYS
            result.weekly_breach = True
            logger.warning(
                "weekly_drawdown_breach: %.4f <= %.2f -> leverage capped at %.2fx "
                "for %d days",
                weekly_drawdown, WEEKLY_DRAWDOWN_LIMIT, REDUCED_LEVERAGE,
                LEVERAGE_REDUCTION_DAYS,
            )

        result.leverage_cap = self.leverage_cap_override
        result.leverage_days_left = self.leverage_override_days_left

        if monthly_drawdown is not None and monthly_drawdown <= MONTHLY_DRAWDOWN_LIMIT:
            for position in self.positions.values():
                position["notional"] *= MONTHLY_SIZE_REDUCTION
            result.monthly_breach = True
            result.positions_reduced = len(self.positions)
            result.alert = {
                "alert_type": "circuit_breaker",
                "data": {
                    "reason": "rolling_month_drawdown",
                    "monthly_drawdown": monthly_drawdown,
                    "action": "reduce_positions_50pct",
                },
            }
            logger.error(
                "monthly_drawdown_breach: %.4f <= %.2f -> reduced %d positions by "
                "%.0f%%, alert queued",
                monthly_drawdown, MONTHLY_DRAWDOWN_LIMIT, result.positions_reduced,
                (1 - MONTHLY_SIZE_REDUCTION) * 100,
            )

        return result

    # ----------------------------------------------------- correlation groups
    def correlated_group(self, symbol: str, regime: str, correlations) -> list[str]:
        """Existing positions effectively correlated > 0.70 with ``symbol``.

        These offer no diversification in the prevailing ``regime`` and are
        treated as the same position for the per-position concentration cap.
        ``correlations`` is a symbol-indexed matrix (e.g. a pandas DataFrame).
        """
        group = []
        for sym in self.positions:
            if sym == symbol:
                continue
            try:
                raw = float(correlations.loc[symbol, sym])
            except (KeyError, TypeError, ValueError):
                continue
            if get_effective_correlation(raw, regime) > CORRELATION_GROUP_THRESHOLD:
                group.append(sym)
        return group

    # -------------------------------------------------------- position sizing
    def calculate_position_size(
        self,
        nav: float,
        entry_price: float,
        stop_price: float,
        *,
        symbol: str | None = None,
        sector: str | None = None,
        regime: str = "neutral",
        correlations=None,
    ) -> float:
        """Return the dollar notional for a new position, respecting all caps.

        Enforces simultaneously: fixed-fractional risk per trade, per-position
        NAV cap, portfolio leverage cap, sector concentration cap, a
        correlation-grouping cap (positions correlated > 0.70 share the
        per-position cap), and the current circuit-breaker size multiplier.
        Returns 0.0 when new positions are not currently allowed or inputs are
        degenerate.
        """
        if not self.allow_new_position():
            return 0.0
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0 or nav <= 0 or entry_price <= 0:
            return 0.0

        # Risk-based size: risk dollars / stop distance => shares => notional.
        risk_dollars = RISK_PER_TRADE_PCT * nav
        shares = risk_dollars / stop_distance
        notional = shares * entry_price

        # Per-position NAV cap.
        notional = min(notional, MAX_POSITION_SIZE_PCT * nav)

        # Correlation-grouping cap: highly-correlated existing positions count
        # toward the SAME per-position cap as the candidate (treat as one).
        if symbol is not None and correlations is not None:
            group_exposure = sum(
                self.positions[sym]["notional"]
                for sym in self.correlated_group(symbol, regime, correlations)
            )
            remaining_group = max(0.0, MAX_POSITION_SIZE_PCT * nav - group_exposure)
            notional = min(notional, remaining_group)

        # Portfolio leverage cap (honors the weekly-drawdown leverage cooldown).
        gross = sum(p["notional"] for p in self.positions.values())
        remaining_leverage = max(0.0, self.effective_max_leverage() * nav - gross)
        notional = min(notional, remaining_leverage)

        # Sector concentration cap.
        if sector is not None:
            sector_exposure = sum(
                p["notional"]
                for p in self.positions.values()
                if p.get("sector") == sector
            )
            remaining_sector = max(0.0, SECTOR_CONCENTRATION_MAX * nav - sector_exposure)
            notional = min(notional, remaining_sector)

        # Circuit-breaker size reduction (e.g. 0.75x at level 1).
        notional *= self.size_multiplier
        return notional

    # ------------------------------------------------------ circuit breakers
    def update(
        self,
        *,
        intraday_drawdown: float,
        drawdown_from_peak: float = 0.0,
        regime: str = "neutral",
        correlations=None,
    ) -> CircuitBreakerResult:
        """Evaluate the staged circuit breakers and apply the matching action.

        ``intraday_drawdown`` and ``drawdown_from_peak`` are signed fractions
        (negative = loss). The most severe applicable level wins; each call
        performs exactly one level's action.
        """
        # Permanent lock takes precedence over everything.
        if self.locked or check_lockfile():
            self.locked = self.block_new = self.halted = True
            return CircuitBreakerResult(
                5, "locked", "Lockfile present; trading halted.",
                block_new=True, halted=True, locked=True,
            )

        # Session halt persists until reset_session().
        if self.halted:
            return CircuitBreakerResult(
                4, "halted", "Session halted.", block_new=True, halted=True
            )

        # Level 5: permanent lockfile halt (uses peak drawdown).
        if drawdown_from_peak <= CIRCUIT_BREAKER_LOCKFILE:
            return self._trigger_lockfile(intraday_drawdown, drawdown_from_peak)

        # Level 4: close all, halt for the session.
        if intraday_drawdown <= CIRCUIT_BREAKER_CLOSE_ALL:
            closed = list(self.positions)
            self.positions.clear()
            self.halted = self.block_new = True
            logger.error(
                "CIRCUIT BREAKER L4: closing all %d positions, halting session "
                "(intraday_drawdown=%.4f)",
                len(closed), intraday_drawdown,
            )
            return CircuitBreakerResult(
                4, "close_all", "Closed all positions; session halted.",
                block_new=True, halted=True, closed_positions=closed,
            )

        # Level 3: close the single weakest-correlated position; recheck later.
        if intraday_drawdown <= CIRCUIT_BREAKER_CLOSE_WEAKEST:
            closed = []
            if self.positions:
                weakest = self._weakest_position(regime, correlations)
                if weakest is not None:
                    self.positions.pop(weakest)
                    closed = [weakest]
            self.block_new = True
            logger.warning(
                "CIRCUIT BREAKER L3: closed weakest-correlated position %s "
                "(intraday_drawdown=%.4f)",
                closed, intraday_drawdown,
            )
            return CircuitBreakerResult(
                3, "close_weakest", f"Closed weakest-correlated: {closed}.",
                block_new=True, closed_positions=closed,
            )

        # Level 2: halve existing positions, block new ones.
        if intraday_drawdown <= CIRCUIT_BREAKER_HALF:
            for p in self.positions.values():
                p["notional"] *= 0.5
            self.block_new = True
            logger.warning(
                "CIRCUIT BREAKER L2: halved all positions, blocking new entries "
                "(intraday_drawdown=%.4f)",
                intraday_drawdown,
            )
            return CircuitBreakerResult(
                2, "halve_and_block", "Halved positions; new entries blocked.",
                block_new=True,
            )

        # Level 1: soft size reduction on new positions.
        if intraday_drawdown <= CIRCUIT_BREAKER_WARN:
            self.size_multiplier = LEVEL1_SIZE_MULTIPLIER
            self.block_new = False
            logger.warning(
                "CIRCUIT BREAKER L1: reducing new-position sizes to %.2fx "
                "(intraday_drawdown=%.4f)",
                LEVEL1_SIZE_MULTIPLIER, intraday_drawdown,
            )
            return CircuitBreakerResult(
                1, "reduce_size", "New-position sizes reduced to 0.75x.",
                size_multiplier=LEVEL1_SIZE_MULTIPLIER,
            )

        # Normal: full size, no blocks.
        self.size_multiplier = 1.0
        self.block_new = False
        return CircuitBreakerResult(0, "normal", "", size_multiplier=1.0)

    # ----------------------------------------------------------------- helpers
    def _weakest_position(self, regime: str, correlations) -> str | None:
        """Symbol with the lowest mean effective correlation to the others."""
        symbols = list(self.positions)
        if not symbols:
            return None
        if correlations is None or len(symbols) < 2:
            return symbols[0]

        scores: dict[str, float] = {}
        for s in symbols:
            others = [t for t in symbols if t != s]
            effective = [
                get_effective_correlation(float(correlations.loc[s, t]), regime)
                for t in others
            ]
            scores[s] = float(np.mean(effective))
        return min(scores, key=scores.get)

    def _trigger_lockfile(
        self, intraday_drawdown: float, drawdown_from_peak: float
    ) -> CircuitBreakerResult:
        """Close all, write the lockfile, dump state, and halt permanently."""
        closed = list(self.positions)
        self.positions.clear()
        self.locked = self.halted = self.block_new = True

        state = {
            "reason": "circuit_breaker_level_5",
            "timestamp": datetime.now().isoformat(),
            "drawdown_from_peak": drawdown_from_peak,
            "intraday_drawdown": intraday_drawdown,
            "closed_positions": closed,
        }
        try:
            LOCKFILE_PATH.write_text(json.dumps(state, indent=2))
        except OSError as exc:  # pragma: no cover - filesystem failure
            logger.error("Failed to write lockfile at %s: %s", LOCKFILE_PATH, exc)
        logger.critical("CIRCUIT BREAKER L5 LOCKFILE — state dump: %s", json.dumps(state))

        return CircuitBreakerResult(
            5, "lockfile",
            "Closed all positions; lockfile written; halted permanently.",
            block_new=True, halted=True, locked=True, closed_positions=closed,
        )
