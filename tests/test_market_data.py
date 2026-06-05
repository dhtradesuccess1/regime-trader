"""Tests for ``data.market_data`` with yfinance mocked (no network)."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from data.market_data import download_ohlcv, download_universe


def _install_fake_yfinance(monkeypatch, frame=None, raises=False):
    """Insert a fake ``yfinance`` module whose download() returns ``frame``."""
    fake = types.ModuleType("yfinance")

    def download(ticker, **kwargs):
        if raises:
            raise RuntimeError("network down")
        return frame

    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _multiindex_frame(n=30):
    idx = pd.date_range("2021-01-04", periods=n, freq="B", name="Date")
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, ["SPY"]])
    data = np.random.default_rng(0).uniform(100, 200, size=(n, len(fields)))
    return pd.DataFrame(data, index=idx, columns=cols)


def test_download_flattens_multiindex_and_selects_ohlcv(monkeypatch):
    _install_fake_yfinance(monkeypatch, frame=_multiindex_frame())
    df = download_ohlcv("SPY", start="2021-01-01", end="2021-03-01")
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert "Adj Close" not in df.columns
    assert len(df) == 30


def test_download_period_path(monkeypatch):
    _install_fake_yfinance(monkeypatch, frame=_multiindex_frame(10))
    df = download_ohlcv("QQQ", period="1y")
    assert len(df) == 10


def test_download_empty_raises(monkeypatch):
    _install_fake_yfinance(monkeypatch, frame=pd.DataFrame())
    with pytest.raises(RuntimeError, match="No data"):
        download_ohlcv("BAD", start="2021-01-01", end="2021-02-01")


def test_download_missing_columns_raises(monkeypatch):
    idx = pd.date_range("2021-01-04", periods=5, freq="B")
    frame = pd.DataFrame({"Open": range(5), "Close": range(5)}, index=idx)
    _install_fake_yfinance(monkeypatch, frame=frame)
    with pytest.raises(RuntimeError, match="missing columns"):
        download_ohlcv("SPY", period="1y")


def test_download_universe_skips_failures(monkeypatch):
    _install_fake_yfinance(monkeypatch, raises=True)
    result = download_universe(["SPY", "QQQ"], period="1y")
    assert result == {}  # all failed -> skipped, no exception
