"""Market data ingestion via yfinance.

Downloads daily OHLCV bars for the configured universe and normalizes them into
a consistent ``Open, High, Low, Close, Volume`` schema indexed by timestamp,
regardless of yfinance quirks (MultiIndex columns for single tickers, an extra
``Adj Close`` column, etc.).

Prices are returned *unadjusted* (``auto_adjust=False``) to match the rest of
the pipeline; pass ``auto_adjust=True`` if you want split/dividend-adjusted
prices instead.
"""

import pandas as pd

from monitoring.logging_config import get_logger

logger = get_logger("market_data")

# The canonical columns every consumer downstream expects.
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def download_ohlcv(
    ticker: str,
    *,
    period: str | None = "2y",
    start=None,
    end=None,
    interval: str = "1d",
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download and normalize OHLCV bars for a single ticker.

    Provide either ``period`` (e.g. ``"2y"``) or an explicit ``start``/``end``
    range; an explicit range takes precedence. Raises ``RuntimeError`` if no
    data is returned (bad symbol, network failure, or empty range).

    Example::

        df = download_ohlcv("SPY", start="2021-01-01", end="2024-01-01")
    """
    import yfinance as yf  # imported lazily so the module loads without network

    use_range = start is not None or end is not None
    logger.info(
        "download_ohlcv",
        ticker=ticker,
        period=None if use_range else period,
        start=str(start) if use_range else None,
        end=str(end) if use_range else None,
        interval=interval,
    )

    if use_range:
        raw = yf.download(
            ticker, start=start, end=end, interval=interval,
            auto_adjust=auto_adjust, progress=False,
        )
    else:
        raw = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=auto_adjust, progress=False,
        )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"No data returned for {ticker!r} "
            f"({'range' if use_range else 'period'} request)."
        )

    # yfinance returns MultiIndex columns (field, ticker) for single tickers in
    # recent versions; flatten to the field level.
    if getattr(raw.columns, "nlevels", 1) > 1:
        raw.columns = raw.columns.get_level_values(0)

    missing = [c for c in OHLCV_COLUMNS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"{ticker!r} response missing columns: {missing}.")

    df = raw[OHLCV_COLUMNS].copy()
    logger.info("download_ohlcv_done", ticker=ticker, n_rows=len(df))
    return df


def download_universe(
    tickers: list[str], *, period: str | None = "2y", start=None, end=None
) -> dict[str, pd.DataFrame]:
    """Download OHLCV for several tickers, returning ``{ticker: DataFrame}``.

    Tickers that fail to download are logged and skipped rather than aborting
    the whole batch.
    """
    out: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            out[ticker] = download_ohlcv(ticker, period=period, start=start, end=end)
        except Exception as exc:
            logger.error("download_failed", ticker=ticker, error=str(exc))
    return out
