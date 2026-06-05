"""Application entry point and orchestration for regime_trader.

Usage::

    python main.py --mode=dry_run     # train on history, replay OOS, place last 5 paper orders
    python main.py --mode=backtest    # full walk-forward backtest over the universe
    python main.py --mode=live        # (not implemented in this build)

The orchestration functions accept injected dependencies (Alpaca client, data
loader, executor) so the full pipeline is unit-testable without network access.
All structured output goes to ``./logs`` as JSONL via structlog; human-readable
summaries go to the console via a console logger (never ``print``).
"""

import argparse
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from alpaca.trading.enums import OrderSide

from broker.alpaca_client import AlpacaClient
from broker.order_executor import OrderExecutor
from core.feature_engineering import compute_features
from core.hmm_engine import HMMRegimeEngine
from core.performance import evaluate_validation_gates
from core.regime_strategies import generate_signal
from core.risk_manager import RiskManager, check_lockfile
from data.data_validator import validate_ohlcv
from data.market_data import download_ohlcv
from monitoring.logging_config import (
    bind_trading_context,
    configure_logging,
    get_console_logger,
    get_logger,
)
from settings.config import HMM_N_REGIMES_RANGE, LOG_DIR_DEFAULT, TICKERS

log = get_logger("main")
console = get_console_logger()

DRY_RUN_TRAIN_FRACTION = 0.80
DRY_RUN_SUBMIT_LAST_N = 5


# --------------------------------------------------------------- environment
def validate_environment() -> None:
    """Load .env and raise a clear error if required credentials are missing."""
    load_dotenv()
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
        raise RuntimeError(
            "Missing Alpaca credentials. Create a .env file with ALPACA_API_KEY "
            "and ALPACA_SECRET_KEY (see .env.example)."
        )


# ----------------------------------------------------------------- dry run
def run_dry_run(
    *,
    client=None,
    data_loader=download_ohlcv,
    executor=None,
    n_regimes_range: tuple[int, int] = HMM_N_REGIMES_RANGE,
    submit_last_n: int = DRY_RUN_SUBMIT_LAST_N,
    do_submit: bool = True,
) -> dict:
    """Run the dry-run pipeline: train, replay OOS, place the last N paper orders."""
    # 2. Validate environment (only on the real path; injected clients skip it).
    if client is None:
        validate_environment()

    # 3. Lockfile gate.
    if check_lockfile():
        console.warning("Lockfile present — trading is halted. Delete lockfile.lock to resume.")
        log.warning("dry_run_halted_lockfile")
        return {"halted": True, "reason": "lockfile"}

    # 4. Alpaca connection.
    if client is None:
        client = AlpacaClient()
    account = client.get_account()
    portfolio_value = float(account.equity)
    log.info("alpaca_connected", portfolio_value=portfolio_value)

    if executor is None:
        executor = OrderExecutor(client.trading_client)
    risk = RiskManager()

    all_signals: list[dict] = []
    breakers: list[int] = []
    last_pred: dict | None = None

    for ticker in TICKERS:
        # 5. Data.
        df = data_loader(ticker)
        # 6. Validate.
        report = validate_ohlcv(df, ticker)
        log.info("data_validated", **{k: report[k] for k in ("ticker", "n_rows", "is_valid")})
        if not report["is_valid"]:
            log.warning("skipping_invalid_ticker", ticker=ticker, issues=report["issues"])
            continue

        # 7. Features.
        feats = compute_features(df)
        if len(feats) < 60:
            log.warning("insufficient_features", ticker=ticker, n=len(feats))
            continue

        # 8. Train HMM on first 80%.
        split = int(len(feats) * DRY_RUN_TRAIN_FRACTION)
        train, oos = feats.iloc[:split], feats.iloc[split:]
        engine = HMMRegimeEngine(n_regimes_range).fit(train)
        engine.reset_online()

        # 9 & 10. Replay OOS bar-by-bar, generate + risk-check signals.
        for date, row in oos.iterrows():
            pred = engine.predict_online(row)
            last_pred = pred
            signal = generate_signal(pred)

            cb = risk.update(intraday_drawdown=0.0, drawdown_from_peak=0.0,
                            regime=signal["regime"])
            if cb.level > 0:
                breakers.append(cb.level)

            close_price = float(df.loc[date, "Close"]) if date in df.index else None
            entry = {
                "ticker": ticker,
                "date": str(date.date() if hasattr(date, "date") else date),
                "close": close_price,
                **signal,
            }
            all_signals.append(entry)
            log.info("dry_run_signal", **entry)

        bind_trading_context(
            regime=last_pred["current_regime"],
            confidence=last_pred["confidence"],
            portfolio_value=portfolio_value,
        )

    # 11. Submit paper orders for the most recent signal PER TICKER (one
    # representative order per symbol), not the global tail of the flat list --
    # otherwise only the last-processed ticker ever gets orders. Capped at
    # submit_last_n as a safety bound.
    latest_by_ticker = {}
    for sig in all_signals:
        latest_by_ticker[sig["ticker"]] = sig  # last write wins => latest bar
    to_submit = list(latest_by_ticker.values())[:submit_last_n]

    orders_placed = []
    if do_submit:
        for sig in to_submit:
            if sig["target_weight"] <= 0 or not sig["close"]:
                log.info("dry_run_skip_flat_signal", ticker=sig["ticker"], date=sig["date"])
                continue
            notional = risk.calculate_position_size(
                portfolio_value, sig["close"], sig["close"] * 0.97
            ) * sig["target_weight"]
            qty = max(1, int(notional / sig["close"])) if sig["close"] else 1
            order_id = executor.submit_market_order(sig["ticker"], qty, OrderSide.BUY)
            if order_id:
                orders_placed.append(order_id)
                log.info("dry_run_order_placed", ticker=sig["ticker"], qty=qty, order_id=order_id)

    # 13. Summary.
    summary = {
        "total_signals": len(all_signals),
        "orders_placed": len(orders_placed),
        "final_regime": last_pred["current_regime"] if last_pred else None,
        "final_confidence": last_pred["confidence"] if last_pred else None,
        "portfolio_value": portfolio_value,
        "circuit_breakers_triggered": sorted(set(breakers)),
    }
    console.info(
        "dry_run_summary",
        total_signals=summary["total_signals"],
        orders_placed=summary["orders_placed"],
        final_regime=summary["final_regime"],
        final_confidence=summary["final_confidence"],
        portfolio_value=summary["portfolio_value"],
        circuit_breakers=summary["circuit_breakers_triggered"],
    )
    log.info("dry_run_complete", **summary)
    return summary


# ------------------------------------------------------------------ backtest
def _aggregate(results: list[dict]) -> dict:
    """Mean of key per-window metrics across all windows."""
    if not results:
        return {}
    keys = [
        "total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
        "win_rate", "benchmark_buy_hold", "benchmark_sma200", "benchmark_random_median",
    ]
    return {f"avg_{k}": sum(r[k] for r in results) / len(results) for k in keys}


def run_backtest(
    *,
    data_loader=download_ohlcv,
    n_regimes_range: tuple[int, int] = HMM_N_REGIMES_RANGE,
    log_dir: str = LOG_DIR_DEFAULT,
) -> dict:
    """Run the full walk-forward backtest over the universe and save results."""
    from pathlib import Path

    from core.backtester import BACKTEST_END, BACKTEST_START, WalkForwardBacktester

    results_by_ticker: dict[str, list[dict]] = {}
    aggregate_by_ticker: dict[str, dict] = {}

    for ticker in TICKERS:
        df = data_loader(ticker, period=None, start=BACKTEST_START, end=BACKTEST_END)
        backtester = WalkForwardBacktester(df, n_regimes_range=n_regimes_range)
        windows = backtester.run()
        results_by_ticker[ticker] = windows
        aggregate_by_ticker[ticker] = _aggregate(windows)
        console.info(
            "backtest_ticker_done",
            ticker=ticker,
            windows=len(windows),
            **aggregate_by_ticker[ticker],
        )
        for i, window in enumerate(windows):
            log.info("backtest_window", ticker=ticker, window=i,
                    sharpe=window["sharpe_ratio"], total_return=window["total_return"])

    # Week-2 validation gates, evaluated on all OOS windows pooled across the
    # universe. A failing gate means iterate on the strategy, not the threshold.
    all_windows = [w for windows in results_by_ticker.values() for w in windows]
    gates = evaluate_validation_gates(all_windows)
    for g in gates["gates"]:
        console.info(
            "validation_gate",
            gate=g["name"],
            passed=g["passed"],
            value=g["value"],
            threshold=g["threshold"],
        )
    console.info("validation_gates_overall", passed=gates["passed"],
                 n_windows=gates["n_windows"])
    log.info("validation_gates", passed=gates["passed"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": TICKERS,
        "results_by_ticker": results_by_ticker,
        "aggregate_by_ticker": aggregate_by_ticker,
        "validation_gates": gates,
    }
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(log_dir) / f"backtest_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    console.info("backtest_saved", path=str(out_path))
    log.info("backtest_complete", path=str(out_path))
    payload["output_path"] = str(out_path)
    return payload


# ---------------------------------------------------------------------- live
def run_live() -> dict:
    """Live trading is intentionally not implemented in this build."""
    console.warning("Live mode is not implemented in this build. Use --mode=dry_run.")
    log.warning("live_mode_not_implemented")
    return {"status": "not_implemented"}


# ----------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="regime_trader")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "backtest", "live"],
        required=True,
        help="Execution mode.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    configure_logging()
    log.info("startup", mode=args.mode)

    if args.mode == "dry_run":
        return run_dry_run()
    if args.mode == "backtest":
        return run_backtest()
    return run_live()


if __name__ == "__main__":  # pragma: no cover
    main()
