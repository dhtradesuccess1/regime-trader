"""Performance analytics and the Week-2 validation gates.

The gates encode the minimum bar a strategy must clear (out-of-sample) before
advancing. They are deliberately strict constants -- the intended response to a
failing gate is to iterate on *strategy parameters*, not to loosen the gate.

Gate mapping to per-window metrics (from ``WalkForwardBacktester``):

1. Sharpe > 0.5 (OOS)            -> mean of per-window ``sharpe_ratio``
2. Max drawdown < 25%            -> mean per-window ``max_drawdown`` magnitude
3. Beats buy & hold > 60% wins   -> windows where ``sharpe_ratio`` > ``benchmark_buy_hold``
4. Beats random > 75% wins       -> windows where ``sharpe_ratio`` > ``benchmark_random_median``
5. >= 3 regimes across period    -> union of ``regime_breakdown`` keys
6. No window drawdown > 40%      -> worst per-window ``max_drawdown`` magnitude
7. Avg cost < 0.15% per trade    -> mean ``avg_slippage_bps`` + commission (bps)

Gates 2 and 6 are intentionally distinct: gate 2 caps the *typical* (average)
window drawdown, gate 6 is a catastrophe guard on the *worst single* window.
"""

import numpy as np

# --- Gate thresholds (do NOT loosen these to pass; fix the strategy) --------
MIN_SHARPE_OOS = 0.5
MAX_AVG_DRAWDOWN = 0.25
MIN_BEAT_BUY_HOLD_PCT = 0.60
MIN_BEAT_RANDOM_PCT = 0.75
MIN_REGIMES_DETECTED = 3
MAX_SINGLE_WINDOW_DRAWDOWN = 0.40
MAX_COST_BPS = 15.0  # 0.15%

# Alpaca US equities are commission-free; kept explicit so the gate accounts for
# it and can be raised for brokers/instruments that do charge commission.
COMMISSION_BPS = 0.0


def _gate(name: str, label: str, value, threshold, passed: bool) -> dict:
    return {
        "name": name,
        "label": label,
        "value": value,
        "threshold": threshold,
        "passed": bool(passed),
    }


def evaluate_validation_gates(
    windows: list[dict], commission_bps: float = COMMISSION_BPS
) -> dict:
    """Evaluate the Week-2 validation gates against per-window backtest results.

    ``windows`` is the list of per-window metric dicts produced by the
    backtester (pooled across the universe is fine). Returns a report with each
    gate's measured value, threshold, and pass flag, plus an overall ``passed``.
    """
    n = len(windows)
    if n == 0:
        return {
            "passed": False,
            "n_windows": 0,
            "reason": "no OOS windows to evaluate",
            "gates": [],
        }

    sharpes = np.array([w["sharpe_ratio"] for w in windows], dtype=float)
    drawdowns = np.array([abs(w["max_drawdown"]) for w in windows], dtype=float)
    slippage_bps = np.array([w["avg_slippage_bps"] for w in windows], dtype=float)

    mean_sharpe = float(sharpes.mean())
    avg_drawdown = float(drawdowns.mean())
    worst_drawdown = float(drawdowns.max())

    beat_bh = float(
        np.mean([w["sharpe_ratio"] > w["benchmark_buy_hold"] for w in windows])
    )
    beat_random = float(
        np.mean([w["sharpe_ratio"] > w["benchmark_random_median"] for w in windows])
    )

    regimes = set()
    for w in windows:
        regimes.update(w.get("regime_breakdown", {}).keys())
    n_regimes = len(regimes)

    avg_cost_bps = float(slippage_bps.mean()) + commission_bps

    gates = [
        _gate("sharpe_oos", "Sharpe ratio > 0.5 (out-of-sample)",
              round(mean_sharpe, 3), MIN_SHARPE_OOS, mean_sharpe > MIN_SHARPE_OOS),
        _gate("max_drawdown", "Average window max drawdown < 25%",
              round(avg_drawdown, 4), MAX_AVG_DRAWDOWN, avg_drawdown < MAX_AVG_DRAWDOWN),
        _gate("beats_buy_hold", "Beats buy & hold on > 60% of windows",
              round(beat_bh, 3), MIN_BEAT_BUY_HOLD_PCT, beat_bh > MIN_BEAT_BUY_HOLD_PCT),
        _gate("beats_random", "Beats random allocation on > 75% of windows",
              round(beat_random, 3), MIN_BEAT_RANDOM_PCT, beat_random > MIN_BEAT_RANDOM_PCT),
        _gate("regimes_detected", "At least 3 distinct regimes across period",
              n_regimes, MIN_REGIMES_DETECTED, n_regimes >= MIN_REGIMES_DETECTED),
        _gate("no_catastrophic_window", "No single window drawdown > 40%",
              round(worst_drawdown, 4), MAX_SINGLE_WINDOW_DRAWDOWN,
              worst_drawdown < MAX_SINGLE_WINDOW_DRAWDOWN),
        _gate("cost_sanity", "Avg slippage + commission < 0.15% (15 bps)",
              round(avg_cost_bps, 3), MAX_COST_BPS, avg_cost_bps < MAX_COST_BPS),
    ]

    return {
        "passed": all(g["passed"] for g in gates),
        "n_windows": n,
        "regimes_detected": sorted(regimes),
        "gates": gates,
    }
