"""Broker integration package for regime_trader.

Wraps connectivity to Alpaca and the mechanics of trading: the API client
(``broker.alpaca_client``), order submission and lifecycle management
(``broker.order_executor``), and live position/exposure tracking
(``broker.position_tracker``). Defaults to Alpaca paper trading.
"""
