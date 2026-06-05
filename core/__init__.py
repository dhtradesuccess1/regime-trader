"""Core quantitative engine for regime_trader.

Contains the building blocks of the strategy: feature engineering, the HMM
regime-detection engine, per-regime strategy logic, the risk manager, the
walk-forward backtester, performance analytics, and stress tests. These modules
are broker-agnostic and operate on price/feature data.
"""
