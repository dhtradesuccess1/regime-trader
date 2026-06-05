# regime_trader

A regime-aware systematic trading system for liquid US ETFs (SPY, QQQ, IWM).

`regime_trader` detects latent market regimes with a Hidden Markov Model (HMM),
maps each regime to a trading strategy, sizes and risk-manages positions, and
executes paper trades through Alpaca. It includes a walk-forward backtester,
stress tests, structured logging, alerting, an LLM-assisted health monitor, and
a Streamlit dashboard.

> **Status:** Scaffold only. No trading logic is implemented yet. Every module
> currently contains a docstring describing its intended contents.

## Project layout

```
regime_trader/
├── main.py                 # Orchestration entry point / scheduled run loop
├── settings/               # Configuration constants and the tradable universe
├── core/                   # Features, HMM engine, strategies, risk, backtest
├── broker/                 # Alpaca connectivity, order execution, positions
├── data/                   # Market data ingestion and validation
├── monitoring/             # Logging, alerts, health monitor, dashboard
└── tests/                  # Unit tests for the core and broker components
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your real credentials. **Never
   commit `.env`** — it is git-ignored.
   ```
   cp .env.example .env
   ```

## Configuration

Trading parameters (universe, HMM settings, risk limits, circuit breakers) live
in `settings/config.py`. Secrets and environment-specific values live in `.env`.

## Disclaimer

This project defaults to Alpaca **paper** trading and is for research and
educational purposes only. It is not financial advice.
