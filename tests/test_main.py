"""Tests for ``main`` orchestration (dry_run, backtest, live, CLI, env)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import core.risk_manager as rm_mod
import main as main_mod
from main import (
    parse_args,
    run_backtest,
    run_dry_run,
    run_live,
    validate_environment,
)


@pytest.fixture(autouse=True)
def isolated_lockfile(tmp_path, monkeypatch):
    monkeypatch.setattr(rm_mod, "LOCKFILE_PATH", tmp_path / "lockfile.lock")


@pytest.fixture(autouse=True)
def single_ticker(monkeypatch):
    """Run over a single ticker to keep HMM training fast and deterministic."""
    monkeypatch.setattr(main_mod, "TICKERS", ["SPY"])


def make_ohlcv(n_rows: int, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n_rows // 2
    returns = np.concatenate(
        [rng.normal(0.0007, 0.007, half), rng.normal(-0.001, 0.02, n_rows - half)]
    )
    close = 400 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n_rows)),
            "High": close * (1 + np.abs(rng.normal(0.003, 0.002, n_rows))),
            "Low": close * (1 - np.abs(rng.normal(0.003, 0.002, n_rows))),
            "Close": close,
            "Volume": rng.integers(50_000_000, 120_000_000, n_rows).astype(float),
        },
        index=pd.date_range("2018-01-02", periods=n_rows, freq="B"),
    )


def make_mock_client(equity=100_000.0, order_id="order-1"):
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity=str(equity))
    client.trading_client.submit_order.return_value = SimpleNamespace(id=order_id)
    return client


# ------------------------------------------------------------------ env / CLI
def test_validate_environment(monkeypatch):
    monkeypatch.setattr(main_mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Alpaca credentials"):
        validate_environment()
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    validate_environment()  # no raise


def test_parse_args_requires_mode():
    with pytest.raises(SystemExit):
        parse_args([])
    assert parse_args(["--mode=backtest"]).mode == "backtest"


def test_run_live_not_implemented():
    assert run_live()["status"] == "not_implemented"


# -------------------------------------------------------------------- dry_run
def test_dry_run_summary(monkeypatch):
    client = make_mock_client()
    loader = lambda ticker, **kw: make_ohlcv(300)
    summary = run_dry_run(
        client=client, data_loader=loader, n_regimes_range=(3, 4), do_submit=False
    )
    assert summary["total_signals"] > 0
    assert summary["final_regime"] is not None
    assert summary["portfolio_value"] == 100_000.0
    assert set(summary) >= {
        "total_signals", "orders_placed", "final_regime",
        "final_confidence", "portfolio_value", "circuit_breakers_triggered",
    }


def test_dry_run_places_orders(monkeypatch):
    # Force every signal to be a tradeable bull so the submit branch runs.
    monkeypatch.setattr(
        main_mod, "generate_signal",
        lambda pred: {"regime": "bull", "confidence": 0.9,
                      "regime_stable": True, "target_weight": 1.0},
    )
    client = make_mock_client(order_id="abc-123")
    loader = lambda ticker, **kw: make_ohlcv(300)
    summary = run_dry_run(
        client=client, data_loader=loader, n_regimes_range=(3, 4), submit_last_n=5
    )
    assert summary["orders_placed"] > 0
    assert client.trading_client.submit_order.called


def test_dry_run_halts_on_lockfile(isolated_lockfile, tmp_path):
    (tmp_path / "lockfile.lock").write_text("locked")
    client = make_mock_client()
    summary = run_dry_run(client=client, data_loader=lambda t, **kw: make_ohlcv(300))
    assert summary == {"halted": True, "reason": "lockfile"}


def test_dry_run_skips_invalid_data(monkeypatch):
    client = make_mock_client()

    def bad_loader(ticker, **kw):
        df = make_ohlcv(300)
        df.loc[df.index[50], "Volume"] = 0  # invalidates the data
        return df

    summary = run_dry_run(
        client=client, data_loader=bad_loader, n_regimes_range=(3, 4), do_submit=False
    )
    # Invalid ticker skipped -> no signals generated.
    assert summary["total_signals"] == 0


# ------------------------------------------------------------------- backtest
def test_run_backtest_saves_results(tmp_path):
    loader = lambda ticker, **kw: make_ohlcv(450)
    payload = run_backtest(
        data_loader=loader, n_regimes_range=(3, 4), log_dir=str(tmp_path)
    )
    assert "SPY" in payload["results_by_ticker"]
    assert len(payload["results_by_ticker"]["SPY"]) >= 4
    assert "avg_sharpe_ratio" in payload["aggregate_by_ticker"]["SPY"]

    import json
    out = tmp_path / payload["output_path"].split("\\")[-1].split("/")[-1]
    assert out.exists()
    json.loads(out.read_text(encoding="utf-8"))  # valid JSON


# ----------------------------------------------------------------- dispatch
def test_main_dispatch(monkeypatch):
    monkeypatch.setattr(main_mod, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "run_dry_run", lambda: {"mode": "dry_run"})
    monkeypatch.setattr(main_mod, "run_backtest", lambda: {"mode": "backtest"})
    monkeypatch.setattr(main_mod, "run_live", lambda: {"mode": "live"})
    assert main_mod.main(["--mode=dry_run"])["mode"] == "dry_run"
    assert main_mod.main(["--mode=backtest"])["mode"] == "backtest"
    assert main_mod.main(["--mode=live"])["mode"] == "live"
