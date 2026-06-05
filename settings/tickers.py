"""Tradable universe definitions and instrument metadata.

This module will describe the set of instruments the system may trade and any
per-ticker metadata needed downstream — for example asset class, exchange,
typical liquidity, tradability flags, and grouping (e.g. broad-market vs.
sector). It complements the ``TICKERS`` constant in ``settings.config`` by
providing richer, structured information and helper accessors for selecting and
validating symbols.

No logic is implemented yet; this is a scaffold.
"""
