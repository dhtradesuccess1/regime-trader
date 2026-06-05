"""Tests for ``core.performance`` validation gates."""

from core.performance import evaluate_validation_gates


def make_window(
    *,
    sharpe=1.0,
    max_drawdown=-0.10,
    bh=0.2,
    rand=0.0,
    regime="bull",
    slippage_bps=5.0,
):
    """A per-window metrics dict with the fields the gates read."""
    return {
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_buy_hold": bh,
        "benchmark_random_median": rand,
        "regime_breakdown": {regime: 1.0},
        "avg_slippage_bps": slippage_bps,
    }


def passing_windows():
    # Strong Sharpe, shallow drawdowns, beats both benchmarks, 3 regimes, cheap.
    regimes = ["bear", "neutral", "bull", "bull"]
    return [make_window(regime=r) for r in regimes]


def gate(report, name):
    return next(g for g in report["gates"] if g["name"] == name)


def test_all_gates_pass():
    report = evaluate_validation_gates(passing_windows())
    assert report["passed"] is True
    assert all(g["passed"] for g in report["gates"])
    assert report["regimes_detected"] == ["bear", "bull", "neutral"]


def test_empty_windows_fails():
    report = evaluate_validation_gates([])
    assert report["passed"] is False
    assert report["n_windows"] == 0


def test_low_sharpe_fails_only_that_gate():
    windows = [make_window(sharpe=0.3, regime=r) for r in ("bear", "neutral", "bull")]
    report = evaluate_validation_gates(windows)
    assert report["passed"] is False
    assert gate(report, "sharpe_oos")["passed"] is False
    # Beating benchmarks still holds (0.3 > 0.2 bh, 0.3 > 0.0 rand).
    assert gate(report, "beats_buy_hold")["passed"] is True


def test_catastrophic_window_fails_gate6_not_gate2():
    # One window with a 45% drawdown; others shallow so the AVERAGE stays < 25%.
    windows = [make_window(regime=r) for r in ("bear", "neutral", "bull")]
    windows.append(make_window(max_drawdown=-0.45, regime="crash"))
    report = evaluate_validation_gates(windows)
    assert gate(report, "no_catastrophic_window")["passed"] is False
    assert gate(report, "max_drawdown")["passed"] is True  # avg still under 25%
    assert report["passed"] is False


def test_fewer_than_three_regimes_fails():
    windows = [make_window(regime="bull") for _ in range(4)]
    report = evaluate_validation_gates(windows)
    assert gate(report, "regimes_detected")["passed"] is False


def test_high_cost_fails_sanity_gate():
    windows = [make_window(slippage_bps=20.0, regime=r) for r in ("bear", "neutral", "bull")]
    report = evaluate_validation_gates(windows)
    assert gate(report, "cost_sanity")["passed"] is False


def test_does_not_beat_random_fails():
    # Strategy Sharpe below the random benchmark on every window.
    windows = [
        make_window(sharpe=0.6, rand=0.9, regime=r) for r in ("bear", "neutral", "bull")
    ]
    report = evaluate_validation_gates(windows)
    assert gate(report, "beats_random")["passed"] is False
