"""Walk-forward backtesting engine with honest, regime-dependent cost modeling.

The backtester is built to avoid the two classic ways a backtest lies:

* **Look-ahead bias.** Regimes are inferred online (forward algorithm, one bar
  at a time) and every decision at bar *t* uses only bars ``0..t``. Each
  walk-forward window trains the HMM strictly on its training slice and then
  trades the out-of-sample slice that follows.
* **Free, frictionless fills.** Costs are *variable*, not a flat constant.
  Slippage scales with regime stress and order size; fills can be partial in
  stressed regimes; large overnight gaps block new entries for the session.

It runs on a single price series (e.g. SPY). A portfolio loop over the universe
can wrap this per-ticker; that is out of scope here.

The regime -> target-weight strategy used here is a simple, transparent
long/flat default (``DEFAULT_TARGET_WEIGHTS``); ``core.regime_strategies`` will
later supply richer logic. It is injectable via the constructor.
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd

from core.feature_engineering import WARMUP_ROWS, compute_features
from core.hmm_engine import HMMRegimeEngine
from settings.config import (
    GAP_OPEN_THRESHOLD,
    HMM_N_REGIMES_RANGE,
    WALK_FORWARD_OOS_DAYS,
    WALK_FORWARD_STEP_DAYS,
    WALK_FORWARD_TRAIN_DAYS,
)

logger = logging.getLogger(__name__)

# Default full-backtest data range (used by real runs, not by the unit tests).
BACKTEST_START = "2018-01-01"
BACKTEST_END = "2024-01-01"

# Minimum number of walk-forward windows required for a valid backtest.
MIN_WINDOWS = 4

# Trading days per year for annualization.
TRADING_DAYS_PER_YEAR = 252

# Number of random strategies used for the random benchmark.
N_RANDOM_BENCHMARK = 100

# --- Cost model: per-regime base slippage (fraction of price) ---------------
# The five values below are the spec's reference rates; the extra two labels
# ("capitulation", "mania") that the HMM may emit at 6-7 regimes extend the
# spectrum to keep cost modeling defined across all possible labels.
REGIME_BASE_SLIPPAGE = {
    "capitulation": 0.0060,
    "crash": 0.0050,
    "bear": 0.0020,
    "neutral": 0.0008,
    "bull": 0.0005,
    "euphoria": 0.0010,
    "mania": 0.0015,
}

# Fallback slippage for any regime label not in the table above (per spec).
DEFAULT_SLIPPAGE = 0.0010

# Regimes treated as "stressed" for partial-fill simulation.
STRESSED_REGIMES = ("capitulation", "crash", "bear")

# Order-size threshold below which fills are full and slippage size impact is
# negligible (as a fraction of average daily volume).
SMALL_ORDER_PCT_ADV = 0.001

# Default long/flat strategy: regime label -> target portfolio weight [0, 1].
DEFAULT_TARGET_WEIGHTS = {
    "capitulation": 0.0,
    "crash": 0.0,
    "bear": 0.0,
    "neutral": 0.5,
    "bull": 1.0,
    "euphoria": 0.5,
    "mania": 0.0,
}


# ============================================================================
# Cost model -- implemented exactly to spec.
# ============================================================================
def calculate_slippage(regime: str, order_size_pct_adv: float) -> float:
    """Variable slippage by regime and order size (NOT a constant).

    Base rate is looked up per regime; the size multiplier grows the cost as the
    order consumes more of average daily volume, capped at 5x base.
    """
    base_rate = REGIME_BASE_SLIPPAGE.get(regime, DEFAULT_SLIPPAGE)
    size_multiplier = 1.0 + min(order_size_pct_adv / 0.001 * 2, 4.0)
    return base_rate * size_multiplier


def simulate_fill_rate(order_size_pct_adv: float, regime: str) -> float:
    """Partial-fill simulation (not always 1.0).

    Tiny orders fill completely; otherwise stressed regimes fill worse than calm
    ones. Reproducibility per window is the caller's responsibility via
    ``np.random.seed(window_index)``.
    """
    if order_size_pct_adv < SMALL_ORDER_PCT_ADV:
        return 1.0
    if regime in STRESSED_REGIMES:
        return float(np.random.uniform(0.55, 0.90))
    return float(np.random.uniform(0.80, 1.00))


# Spec alias: ``simulate_fill`` is the same function under the name used in the
# order-simulation snippet.
simulate_fill = simulate_fill_rate


def check_gap_open(open_price: float, prev_close: float) -> bool:
    """True if the absolute overnight gap exceeds ``GAP_OPEN_THRESHOLD``.

    When True, the caller skips all new entries for that session.
    """
    if prev_close == 0:
        return False
    return abs(open_price / prev_close - 1.0) > GAP_OPEN_THRESHOLD


# ============================================================================
# Helpers
# ============================================================================
def _sharpe(returns) -> float:
    """Annualized Sharpe of a return series; 0.0 if undefined."""
    arr = np.asarray(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


# ============================================================================
# Walk-forward backtester
# ============================================================================
class WalkForwardBacktester:
    """Rolling train/out-of-sample backtester over a single OHLCV series."""

    def __init__(
        self,
        price_df: pd.DataFrame,
        *,
        train_days: int = WALK_FORWARD_TRAIN_DAYS,
        oos_days: int = WALK_FORWARD_OOS_DAYS,
        step_days: int = WALK_FORWARD_STEP_DAYS,
        min_windows: int = MIN_WINDOWS,
        target_weights: dict | None = None,
        n_regimes_range: tuple[int, int] = HMM_N_REGIMES_RANGE,
    ):
        self.price_df = price_df
        self.train_days = train_days
        self.oos_days = oos_days
        self.step_days = step_days
        self.min_windows = min_windows
        self.target_weights = target_weights or dict(DEFAULT_TARGET_WEIGHTS)
        self.n_regimes_range = n_regimes_range

    # --------------------------------------------------------------- windows
    def _window_starts(self) -> list[int]:
        """Integer start offsets for each walk-forward window.

        Raises ValueError if fewer than ``min_windows`` windows are available.
        """
        n = len(self.price_df)
        span = self.train_days + self.oos_days
        starts = list(range(0, n - span + 1, self.step_days))
        if len(starts) < self.min_windows:
            raise ValueError(
                f"Insufficient data for walk-forward: only {len(starts)} "
                f"window(s) available, need at least {self.min_windows}. "
                f"Have {n} bars; each window needs {span} bars "
                f"(train {self.train_days} + OOS {self.oos_days})."
            )
        return starts

    # ------------------------------------------------------------------- run
    def run(self) -> list[dict]:
        """Run all walk-forward windows and return a per-window metrics dict."""
        starts = self._window_starts()
        logger.info("Walk-forward: %d windows", len(starts))
        results = []
        for w, t in enumerate(starts):
            train_df = self.price_df.iloc[t : t + self.train_days]
            # OOS slice plus a warmup buffer so OOS features are causal.
            oos_warmup = self.price_df.iloc[
                t + self.train_days - WARMUP_ROWS : t + self.train_days + self.oos_days
            ]
            oos_df = self.price_df.iloc[
                t + self.train_days : t + self.train_days + self.oos_days
            ]
            results.append(self._run_window(w, train_df, oos_warmup, oos_df))
        return results

    def _run_window(
        self,
        window_index: int,
        train_df: pd.DataFrame,
        oos_warmup_df: pd.DataFrame,
        oos_df: pd.DataFrame,
    ) -> dict:
        engine = HMMRegimeEngine(self.n_regimes_range)
        engine.fit(compute_features(train_df))
        sim = self.simulate_window(engine, oos_warmup_df, window_index)
        return self._metrics(sim, oos_df, window_index)

    # -------------------------------------------------------------- strategy
    def apply_gap_filter(
        self, target_weight: float, current_weight: float, open_price: float, prev_close: float
    ) -> float:
        """Apply the gap-open session policy to a target weight.

        On a large opening gap: do not enter new positions / increase exposure,
        AND reduce existing exposure by 50%. The resulting weight is therefore
        ``min(target, current * 0.5)`` -- this blocks increases (a flat book
        stays flat) while actively cutting any existing position in half. With
        no gap, the target is honored unchanged.
        """
        if check_gap_open(open_price, prev_close):
            return min(target_weight, current_weight * 0.5)
        return target_weight

    # ------------------------------------------------------------- simulate
    def simulate_window(
        self, engine: HMMRegimeEngine, oos_warmup_df: pd.DataFrame, window_index: int
    ) -> dict:
        """Run the OOS bars one at a time and record the trade/equity path.

        Features are recomputed causally from ``oos_warmup_df`` (warmup buffer +
        OOS bars). Each bar's decision uses only data up to that bar.
        """
        feats = compute_features(oos_warmup_df)
        engine.reset_online()
        # Reproducible partial fills per window (spec requirement).
        np.random.seed(window_index)

        opens = oos_warmup_df["Open"]
        closes = oos_warmup_df["Close"]
        adv_dollar = (oos_warmup_df["Close"] * oos_warmup_df["Volume"]).rolling(20).mean()

        equity = 1.0
        current_w = 0.0
        entry_equity = None
        entry_bar = None

        weights, regimes, returns, equity_curve = [], [], [], []
        slippages, fills, trades = [], [], []
        # Unfilled remainder of each order (intended minus executed exposure),
        # tracked separately so partial fills are visible, not silently dropped.
        unfilled_remainder = []

        for i, date in enumerate(feats.index):
            pred = engine.predict_online(feats.loc[date])
            regime = pred["current_regime"]
            stable = pred["regime_stable"]

            # Stable regime sets a new target; otherwise hold the current weight.
            target = self.target_weights.get(regime, 0.0) if stable else current_w

            p = oos_warmup_df.index.get_loc(date)
            open_d = opens.iloc[p]
            prev_close = closes.iloc[p - 1]
            close_d = closes.iloc[p]

            if check_gap_open(open_d, prev_close):
                gap_pct = abs(open_d / prev_close - 1.0)
                logger.warning(
                    "gap_open_detected: gap_pct=%.4f action=reduce_and_block", gap_pct
                )
            desired = self.apply_gap_filter(target, current_w, open_d, prev_close)
            delta = desired - current_w

            cost = 0.0
            w_before = current_w
            if abs(delta) > 1e-9:
                notional = abs(delta) * equity
                adv = adv_dollar.iloc[p]
                size_pct_adv = notional / adv if adv and adv > 0 else 0.0
                slip = calculate_slippage(regime, size_pct_adv)
                fill = simulate_fill_rate(size_pct_adv, regime)
                # actual = intended * fill_rate; the rest is left unfilled.
                executed = delta * fill
                current_w = current_w + executed
                cost = slip * abs(executed)
                slippages.append(slip * 1e4)  # bps
                fills.append(fill)
                unfilled_remainder.append(abs(delta) * (1.0 - fill))

            # Mark to market over the day at the (post-trade) weight, net of cost.
            daily_ret = close_d / prev_close - 1.0
            strat_ret = current_w * daily_ret - cost
            equity *= 1.0 + strat_ret

            # Trade (round-trip) bookkeeping for long/flat positions.
            if w_before == 0.0 and current_w > 0.0:
                entry_equity = equity
                entry_bar = i
            elif w_before > 0.0 and current_w == 0.0 and entry_equity is not None:
                trades.append(
                    {"pnl": equity / entry_equity - 1.0, "hold_bars": i - entry_bar}
                )
                entry_equity = None
                entry_bar = None

            weights.append(current_w)
            regimes.append(regime)
            returns.append(strat_ret)
            equity_curve.append(equity)

        # Close any position still open at the end of the window.
        if entry_equity is not None:
            trades.append(
                {
                    "pnl": equity / entry_equity - 1.0,
                    "hold_bars": len(feats) - 1 - entry_bar,
                }
            )

        return {
            "index": list(feats.index),
            "weights": weights,
            "regimes": regimes,
            "returns": returns,
            "equity_curve": equity_curve,
            "slippage_bps": slippages,
            "fill_rates": fills,
            "unfilled_remainder": unfilled_remainder,
            "trades": trades,
        }

    # --------------------------------------------------------------- metrics
    def _metrics(self, sim: dict, oos_df: pd.DataFrame, window_index: int) -> dict:
        returns = np.asarray(sim["returns"], dtype=float)
        equity_curve = np.asarray(sim["equity_curve"], dtype=float)
        n = len(returns)

        total_return = float(equity_curve[-1] - 1.0) if n else 0.0
        annualized_return = (
            float(equity_curve[-1] ** (TRADING_DAYS_PER_YEAR / n) - 1.0) if n else 0.0
        )
        sharpe = _sharpe(returns)

        if n:
            running_max = np.maximum.accumulate(equity_curve)
            max_drawdown = float((equity_curve / running_max - 1.0).min())
        else:
            max_drawdown = 0.0

        trades = sim["trades"]
        total_trades = len(trades)
        win_rate = (
            float(np.mean([t["pnl"] > 0 for t in trades])) if total_trades else 0.0
        )
        avg_hold_bars = (
            float(np.mean([t["hold_bars"] for t in trades])) if total_trades else 0.0
        )

        regime_counts = Counter(sim["regimes"])
        regime_breakdown = (
            {label: count / n for label, count in regime_counts.items()} if n else {}
        )

        avg_slippage_bps = (
            float(np.mean(sim["slippage_bps"])) if sim["slippage_bps"] else 0.0
        )
        avg_fill_rate = float(np.mean(sim["fill_rates"])) if sim["fill_rates"] else 1.0
        # Partial-fill visibility: total unfilled exposure and how many orders
        # did not fully fill.
        total_unfilled = float(np.sum(sim["unfilled_remainder"]))
        partial_fill_count = int(sum(1 for f in sim["fill_rates"] if f < 1.0))

        bh, sma200, rand_median = self._benchmarks(oos_df)

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_hold_bars": avg_hold_bars,
            "total_trades": total_trades,
            "regime_breakdown": regime_breakdown,
            "avg_slippage_bps": avg_slippage_bps,
            "avg_fill_rate": avg_fill_rate,
            "total_unfilled": total_unfilled,
            "partial_fill_count": partial_fill_count,
            "benchmark_buy_hold": bh,
            "benchmark_sma200": sma200,
            "benchmark_random_median": rand_median,
        }

    # ------------------------------------------------------------ benchmarks
    def _benchmarks(self, oos_df: pd.DataFrame) -> tuple[float, float, float]:
        """All three benchmarks, computed on the identical OOS window.

        Returns Sharpe ratios for buy-and-hold, an SMA200 long/flat rule, and
        the median of 100 random long/flat strategies.
        """
        closes = self.price_df["Close"]
        oos_dates = oos_df.index
        full_ret = closes.pct_change()
        oos_ret = full_ret.loc[oos_dates].to_numpy(dtype=float)

        # Buy & hold.
        bh = _sharpe(oos_ret)

        # SMA200 long/flat, using the prior day's signal (no look-ahead).
        sma = closes.rolling(200).mean()
        signal = (closes > sma).astype(float).shift(1)
        sma_ret = signal.loc[oos_dates].to_numpy(dtype=float) * oos_ret
        sma200 = _sharpe(sma_ret)

        # Random long/flat: 100 seeds, median Sharpe.
        n = len(oos_ret)
        sharpes = []
        for seed in range(N_RANDOM_BENCHMARK):
            np.random.seed(100_000 + seed)
            pos = np.random.randint(0, 2, size=n).astype(float)
            sharpes.append(_sharpe(pos * oos_ret))
        rand_median = float(np.median(sharpes)) if sharpes else 0.0

        return bh, sma200, rand_median
