"""Central configuration constants for regime_trader.

This module holds tunable, non-secret parameters that govern the trading
universe, the HMM regime model, the walk-forward backtest schedule, position
sizing and leverage limits, intraday circuit breakers, regime-stability
filters, gap handling, and bar size. Secrets (API keys, URLs) are *not* defined
here — those come from the environment via ``.env``.

These values are the single source of truth for the system's risk and model
parameters; modules should import from here rather than redefining literals.
"""

# --- Trading universe -------------------------------------------------------
TICKERS = ["SPY", "QQQ", "IWM"]

# --- HMM regime model -------------------------------------------------------
HMM_N_REGIMES_RANGE = (3, 7)
HMM_TRAIN_DAYS = 252

# --- Walk-forward backtest schedule (in trading days) -----------------------
WALK_FORWARD_TRAIN_DAYS = 252
WALK_FORWARD_OOS_DAYS = 63
WALK_FORWARD_STEP_DAYS = 21

# --- Position sizing and leverage -------------------------------------------
MAX_POSITION_SIZE_PCT = 0.15
MAX_PORTFOLIO_LEVERAGE = 1.25
RISK_PER_TRADE_PCT = 0.01

# --- Intraday circuit breakers (fraction of equity drawdown) ----------------
CIRCUIT_BREAKER_WARN = -0.01
CIRCUIT_BREAKER_HALF = -0.02
CIRCUIT_BREAKER_CLOSE_WEAKEST = -0.03
CIRCUIT_BREAKER_CLOSE_ALL = -0.05
CIRCUIT_BREAKER_LOCKFILE = -0.10

# --- Regime stability filters -----------------------------------------------
REGIME_STABILITY_MIN_BARS = 3
REGIME_STABILITY_MAX_FLIPS = 4

# --- Gap handling -----------------------------------------------------------
GAP_OPEN_THRESHOLD = 0.02

# --- Bar size ---------------------------------------------------------------
BAR_SIZE = "1Day"

# --- Operational defaults ---------------------------------------------------
# Default log directory (overridden by the LOG_DIR environment variable).
LOG_DIR_DEFAULT = "./logs"
