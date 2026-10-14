"""Backtester defaults."""

DEFAULT_SYMBOL = "AAPL"
DEFAULT_YEARS = 2
DEFAULT_STRATEGY = "EMA 9/21 crossover"
DEFAULT_STOP_PCT = 5.0
DEFAULT_COMMISSION = 0.0          # $ per side (most brokers are $0 now)
DEFAULT_SLIPPAGE_PCT = 0.05       # % per fill (market-order slippage)
DEFAULT_CAPITAL = 100_000.0
DEFAULT_POSITION = 10_000.0       # fixed $ per trade (no compounding)
ALPACA_FEED = "sip"
