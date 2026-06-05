"""Feature engineering for regime detection and strategy signals.

Transforms raw OHLCV bars into the fixed feature set consumed by the HMM
engine. The feature definitions are intentionally simple and transparent:

1. ``log_return``        -- log of close-to-close return
2. ``realized_vol_5d``   -- 5-bar annualized realized volatility
3. ``realized_vol_20d``  -- 20-bar annualized realized volatility
4. ``volume_ratio``      -- volume relative to its 20-bar mean
5. ``range_pct``         -- bar range as a fraction of the prior close

No scaling/standardization is applied here -- that is the HMM engine's job.
The first 20 rows (warmup for the rolling windows) are dropped.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns required on the input OHLCV frame.
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# The exact feature columns produced (everything else is dropped).
FEATURE_COLUMNS = [
    "log_return",
    "realized_vol_5d",
    "realized_vol_20d",
    "volume_ratio",
    "range_pct",
]

# Number of leading rows discarded so the rolling windows are fully warmed up.
WARMUP_ROWS = 20

# Annualization factor for daily realized volatility.
TRADING_DAYS_PER_YEAR = 252

# Warn if more than this fraction of output rows still contain a NaN.
NAN_WARN_THRESHOLD = 0.02


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the regime feature set from an OHLCV DataFrame.

    Input: OHLCV DataFrame with columns Open, High, Low, Close, Volume.
    Output: DataFrame with exactly the five feature columns (all others
    dropped), with the first 20 (warmup) rows removed.

    Features:

    1. ``log_return``:      ``np.log(close / close.shift(1))``
    2. ``realized_vol_5d``: ``log_return.rolling(5).std() * sqrt(252)``
    3. ``realized_vol_20d``:``log_return.rolling(20).std() * sqrt(252)``
    4. ``volume_ratio``:    ``volume / volume.rolling(20).mean()``
    5. ``range_pct``:       ``(high - low) / close.shift(1)``

    Rules:

    * Drop the first 20 rows (warmup period).
    * No scaling is applied here -- scaling happens in the HMM engine.
    * Raise ``ValueError`` if any required column is missing from the input.
    * Log a warning if more than 2% of output rows contain a NaN.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}."
        )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    log_return = np.log(close / close.shift(1))

    features = pd.DataFrame(index=df.index)
    features["log_return"] = log_return
    features["realized_vol_5d"] = log_return.rolling(5).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    features["realized_vol_20d"] = log_return.rolling(20).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )
    features["volume_ratio"] = volume / volume.rolling(20).mean()
    features["range_pct"] = (high - low) / close.shift(1)

    # Keep only the feature columns, in the canonical order.
    features = features[FEATURE_COLUMNS]

    # Drop the warmup period: the leading rows where rolling windows are not
    # yet fully populated.
    features = features.iloc[WARMUP_ROWS:]

    # Data-quality check on the result.
    n_rows = len(features)
    if n_rows:
        nan_rows = int(features.isna().any(axis=1).sum())
        nan_fraction = nan_rows / n_rows
        if nan_fraction > NAN_WARN_THRESHOLD:
            logger.warning(
                "Feature matrix contains NaNs in %d/%d rows (%.2f%%), "
                "above the %.0f%% threshold.",
                nan_rows,
                n_rows,
                nan_fraction * 100,
                NAN_WARN_THRESHOLD * 100,
            )

    return features
