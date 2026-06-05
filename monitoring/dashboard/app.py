"""Streamlit monitoring dashboard.

This module will provide the operator-facing dashboard for regime_trader,
built with ``streamlit`` and ``plotly``. Planned views:

* Current regime, regime-probability history, and recent regime flips.
* Live positions, exposure/leverage, and equity curve.
* Performance metrics from ``core.performance`` and recent trade ledger.
* Circuit-breaker / risk status and the latest health-monitor summary.

Run (once implemented) with ``streamlit run monitoring/dashboard/app.py``.
No logic is implemented yet; this is a scaffold.
"""
