"""Market data validation and cleaning.

Guards the pipeline against bad input before data reaches feature engineering
and the HMM. :func:`validate_ohlcv` returns a structured quality report and
never mutates the input; :func:`is_valid` is a convenience boolean.

Checks performed:

* Required OHLCV columns present.
* NaN values in any column.
* Non-positive volume (``Volume <= 0``).
* Non-positive prices.
* Duplicate or non-monotonic timestamps.
* Calendar gaps (suspiciously long jumps between consecutive bars).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# A jump larger than this many calendar days between consecutive bars is a gap
# (covers normal weekends/holidays without false positives).
GAP_DAYS_THRESHOLD = 5


def validate_ohlcv(df: pd.DataFrame, ticker: str | None = None) -> dict:
    """Validate an OHLCV DataFrame and return a quality report.

    The report dict contains ``ticker``, ``n_rows``, ``is_valid`` (no critical
    issues), and the individual counts plus a human-readable ``issues`` list.
    """
    issues: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        # Without the columns we cannot run the remaining checks.
        report = {
            "ticker": ticker,
            "n_rows": len(df),
            "is_valid": False,
            "missing_columns": missing,
            "issues": [f"missing columns: {missing}"],
        }
        logger.warning("data_validation_failed", extra={"report": report})
        return report

    n_rows = len(df)
    nan_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    nonpositive_volume = int((df["Volume"] <= 0).sum())
    nonpositive_price = int((df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum())
    duplicate_timestamps = int(df.index.duplicated().sum())
    monotonic = bool(df.index.is_monotonic_increasing)

    # Calendar gaps between consecutive bars.
    gap_count = 0
    if n_rows > 1 and isinstance(df.index, pd.DatetimeIndex):
        deltas = df.index.to_series().diff().dropna()
        gap_count = int((deltas > pd.Timedelta(days=GAP_DAYS_THRESHOLD)).sum())

    if nan_rows:
        issues.append(f"{nan_rows} rows contain NaN")
    if nonpositive_volume:
        issues.append(f"{nonpositive_volume} rows have volume <= 0")
    if nonpositive_price:
        issues.append(f"{nonpositive_price} rows have a price <= 0")
    if duplicate_timestamps:
        issues.append(f"{duplicate_timestamps} duplicate timestamps")
    if not monotonic:
        issues.append("timestamps are not monotonically increasing")
    if gap_count:
        issues.append(f"{gap_count} calendar gaps > {GAP_DAYS_THRESHOLD} days")

    # Critical issues invalidate the data; gaps alone are a warning only.
    is_valid = not (
        nan_rows
        or nonpositive_volume
        or nonpositive_price
        or duplicate_timestamps
        or not monotonic
    )

    report = {
        "ticker": ticker,
        "n_rows": n_rows,
        "is_valid": is_valid,
        "nan_rows": nan_rows,
        "nonpositive_volume": nonpositive_volume,
        "nonpositive_price": nonpositive_price,
        "duplicate_timestamps": duplicate_timestamps,
        "monotonic": monotonic,
        "gap_count": gap_count,
        "issues": issues,
    }
    if not is_valid:
        logger.warning("data_validation_issues: %s (%s)", issues, ticker)
    return report


def is_valid(df: pd.DataFrame, ticker: str | None = None) -> bool:
    """Convenience wrapper returning only the validity boolean."""
    return validate_ohlcv(df, ticker)["is_valid"]
