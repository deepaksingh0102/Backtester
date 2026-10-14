"""Strategies: exact, testable rules.

Each strategy returns a dict with:
  entry: bool Series — True on a bar whose CLOSE generates an entry signal
  exit:  bool Series — True on a bar whose CLOSE generates an exit signal
  max_holding_bars: int | None — if set, the position exits at the close of
    the entry bar + max_holding_bars - 1 (time-based exit, e.g. 1-day hold)
  describe: one-line human-readable rule text

Execution (in engine.py): entry signals fill at the NEXT bar's open, exit
signals fill at the NEXT bar's open, stops fill intrabar at the stop price.
Signals use only data through the signal bar's close — no lookahead.
"""
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (exponential smoothing, alpha = 1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False,
                        min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False,
                        min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ema_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> dict:
    fast_e = ema(df["Close"], fast)
    slow_e = ema(df["Close"], slow)
    entry = (fast_e.shift(1) <= slow_e.shift(1)) & (fast_e > slow_e)
    exit_ = (fast_e.shift(1) >= slow_e.shift(1)) & (fast_e < slow_e)
    return {
        "entry": entry.fillna(False), "exit": exit_.fillna(False),
        "max_holding_bars": None,
        "describe": (f"Buy when EMA {fast} crosses above EMA {slow} "
                     f"(close of signal bar, fill next open). Exit when "
                     f"EMA {fast} crosses below EMA {slow}, or on stop."),
    }


def rsi_reversion(df: pd.DataFrame, period: int = 14, oversold: float = 30,
                  exit_level: float = 50) -> dict:
    r = rsi(df["Close"], period)
    entry = (r.shift(1) < oversold) & (r >= oversold)
    exit_ = (r.shift(1) < exit_level) & (r >= exit_level)
    return {
        "entry": entry.fillna(False), "exit": exit_.fillna(False),
        "max_holding_bars": None,
        "describe": (f"Buy when RSI({period}) recovers back above {oversold:g} "
                     f"after dipping below it (close of signal bar, fill next "
                     f"open). Exit when RSI crosses above {exit_level:g}, "
                     f"or on stop."),
    }


def prior_high_breakout(df: pd.DataFrame) -> dict:
    """Daily analog of the analyzer's opening-range breakout: the prior
    day's high is the range. Buy a close above it, hold 1 day."""
    entry = df["Close"] > df["High"].shift(1)
    exit_ = pd.Series(False, index=df.index)
    return {
        "entry": entry.fillna(False), "exit": exit_,
        "max_holding_bars": 1,
        "describe": ("Buy when the close breaks above the prior day's high "
                     "(fill next open). Exit at the close of the entry day "
                     "(1-day hold), or on stop."),
    }


STRATEGIES = {
    "EMA 9/21 crossover": ema_crossover,
    "RSI mean-reversion": rsi_reversion,
    "Prior-day high breakout": prior_high_breakout,
}
