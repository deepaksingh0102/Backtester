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


def true_range(high: pd.Series, low: pd.Series,
               close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat([high - low, (high - prev_close).abs(),
                      (low - prev_close).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Wilder's ATR (exponential smoothing, alpha = 1/period)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False,
                  min_periods=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Wilder's ADX: trend strength 0-100. Above ~25 = a real trend;
    below ~20 = choppy / ranging. Uses only data through each bar."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = true_range(high, low, close)
    smooth = lambda s: s.ewm(alpha=1 / period, adjust=False,
                              min_periods=period).mean()
    atr_v = smooth(tr)
    plus_di = 100 * smooth(plus_dm) / atr_v
    minus_di = 100 * smooth(minus_dm) / atr_v
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / period, adjust=False,
                  min_periods=period).mean()


def _adx_filter(df: pd.DataFrame, entry: pd.Series,
                adx_min: float | None) -> pd.Series:
    """If adx_min is set, entries only fire when ADX (at the signal bar's
    close) is at or above it. The ADX value is known at the close, so
    filtering on it is not lookahead. NaN ADX (warm-up, ~26 bars — trend
    strength not yet measurable) blocks entries: the filter means 'only
    trade when trend strength is CONFIRMED', and during warm-up it is
    not confirmed."""
    if adx_min is None:
        return entry
    a = adx(df["High"], df["Low"], df["Close"])
    return entry & (a >= adx_min)


def ema_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21,
                  adx_min: float | None = None) -> dict:
    fast_e = ema(df["Close"], fast)
    slow_e = ema(df["Close"], slow)
    entry = (fast_e.shift(1) <= slow_e.shift(1)) & (fast_e > slow_e)
    exit_ = (fast_e.shift(1) >= slow_e.shift(1)) & (fast_e < slow_e)
    entry = _adx_filter(df, entry, adx_min)
    desc = (f"Buy when EMA {fast} crosses above EMA {slow} "
            f"(close of signal bar, fill next open). Exit when "
            f"EMA {fast} crosses below EMA {slow}, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def rsi_reversion(df: pd.DataFrame, period: int = 14, oversold: float = 30,
                  exit_level: float = 50,
                  adx_min: float | None = None) -> dict:
    r = rsi(df["Close"], period)
    entry = (r.shift(1) < oversold) & (r >= oversold)
    exit_ = (r.shift(1) < exit_level) & (r >= exit_level)
    entry = _adx_filter(df, entry, adx_min)
    desc = (f"Buy when RSI({period}) recovers back above {oversold:g} "
            f"after dipping below it (close of signal bar, fill next "
            f"open). Exit when RSI crosses above {exit_level:g}, "
            f"or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def prior_high_breakout(df: pd.DataFrame, vol_mult: float | None = None,
                        vol_window: int = 20,
                        adx_min: float | None = None) -> dict:
    """Daily analog of the analyzer's opening-range breakout: the prior
    day's high is the range. Buy a close above it, hold 1 day."""
    entry = df["Close"] > df["High"].shift(1)
    if vol_mult is not None:
        # Volume must beat a multiple of the PRIOR-window average (the
        # current bar is excluded from its own average — no lookahead).
        avg_vol = (df["Volume"].rolling(vol_window, min_periods=vol_window)
                   .mean().shift(1))
        entry = entry & (df["Volume"] > vol_mult * avg_vol)
    entry = _adx_filter(df, entry, adx_min)
    exit_ = pd.Series(False, index=df.index)
    desc = ("Buy when the close breaks above the prior day's high "
            "(fill next open). Exit at the close of the entry day "
            "(1-day hold), or on stop.")
    if vol_mult is not None:
        desc += (f" Breakout bar volume must exceed {vol_mult:g}x the "
                 f"prior {vol_window}-day average.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_,
            "max_holding_bars": 1, "describe": desc}


STRATEGIES = {
    "EMA 9/21 crossover": ema_crossover,
    "RSI mean-reversion": rsi_reversion,
    "Prior-day high breakout": prior_high_breakout,
}
