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


def vwap(close: pd.Series, high: pd.Series, low: pd.Series,
         volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling volume-weighted average price: sum(typical price x volume)
    / sum(volume) over the window. On daily bars this is a rolling VWAP
    (true session VWAP is intraday); the strategy text says so."""
    tp = (high + low + close) / 3
    tpv = tp * volume
    return (tpv.rolling(window, min_periods=window).sum()
            / volume.rolling(window, min_periods=window).sum())


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple:
    """Returns (MACD line, signal line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20,
              num_std: float = 2.0) -> tuple:
    """Returns (middle SMA, upper band, lower band). Uses population
    standard deviation (Bollinger's own definition)."""
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + num_std * sd, mid - num_std * sd


def keltner_bands(close: pd.Series, high: pd.Series, low: pd.Series,
                  ema_period: int = 20, atr_mult: float = 2.0,
                  atr_period: int = 14) -> tuple:
    """Keltner Channels: EMA +/- multiple x ATR. Returns
    (middle EMA, upper band, lower band)."""
    mid = ema(close, ema_period)
    a = atr(high, low, close, atr_period)
    return mid, mid + atr_mult * a, mid - atr_mult * a


def _volume_filter(df: pd.DataFrame, entry: pd.Series,
                   vol_mult: float | None, vol_window: int = 20) -> pd.Series:
    """If vol_mult is set, entries only fire when the signal bar's volume
    beats vol_mult x the PRIOR-window average volume (the current bar is
    excluded from its own average — no lookahead)."""
    if vol_mult is None:
        return entry
    avg_vol = (df["Volume"].rolling(vol_window, min_periods=vol_window)
               .mean().shift(1))
    return entry & (df["Volume"] > vol_mult * avg_vol)


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
                  adx_min: float | None = None,
                  vol_mult: float | None = None,
                  vol_window: int = 20) -> dict:
    fast_e = ema(df["Close"], fast)
    slow_e = ema(df["Close"], slow)
    entry = (fast_e.shift(1) <= slow_e.shift(1)) & (fast_e > slow_e)
    exit_ = (fast_e.shift(1) >= slow_e.shift(1)) & (fast_e < slow_e)
    entry = _volume_filter(df, entry, vol_mult, vol_window)
    entry = _adx_filter(df, entry, adx_min)
    desc = (f"Buy when EMA {fast} crosses above EMA {slow} "
            f"(close of signal bar, fill next open). Exit when "
            f"EMA {fast} crosses below EMA {slow}, or on stop.")
    if vol_mult is not None:
        desc += (f" Crossover bar volume must exceed {vol_mult:g}x the "
                 f"prior {vol_window}-day average.")
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
    entry = _volume_filter(df, entry, vol_mult, vol_window)
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


def rsi_macd(df: pd.DataFrame, oversold: float = 30,
             exit_level: float = 50, macd_fast: int = 12,
             macd_slow: int = 26, macd_signal: int = 9,
             confirm_window: int = 3,
             adx_min: float | None = None) -> dict:
    """RSI timing + MACD trend confirmation: buy on the first bar where
    BOTH are confirmed within `confirm_window` bars of each other — an RSI
    recovery above oversold AND a positive MACD histogram. Either signal
    may come first. Every value used is known at that bar's close, so
    there is no lookahead. confirm_window=0 reproduces the strict
    same-bar version. MACD confirms the entry only; exit on RSI."""
    r = rsi(df["Close"])
    _, _, hist = macd(df["Close"], macd_fast, macd_slow, macd_signal)
    recovery = (r.shift(1) < oversold) & (r >= oversold)
    hist_pos = (hist > 0).fillna(False)
    w = confirm_window + 1  # window of bars, including the current bar
    rsi_recent = recovery.rolling(w, min_periods=1).max() > 0
    hist_recent = hist_pos.rolling(w, min_periods=1).max() > 0
    both = rsi_recent & hist_recent
    entry = both & ~both.shift(1).fillna(False)  # first bar of each episode
    exit_ = (r.shift(1) < exit_level) & (r >= exit_level)
    entry = _adx_filter(df, entry, adx_min)
    desc = (f"Buy when an RSI recovery above {oversold:g} and a positive "
            f"MACD histogram (MACD {macd_fast}/{macd_slow}/{macd_signal}) "
            f"both occur within {confirm_window} bars of each other. Exit "
            f"when RSI crosses above {exit_level:g}, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def vwap_volume(df: pd.DataFrame, vwap_window: int = 20,
                vol_mult: float | None = 1.5, vol_window: int = 20,
                adx_min: float | None = None) -> dict:
    """Rolling-VWAP crossover with volume confirmation: buy when the close
    crosses above the rolling VWAP on volume above Nx the prior average.
    Exit when the close crosses back below VWAP."""
    vw = vwap(df["Close"], df["High"], df["Low"], df["Volume"], vwap_window)
    entry = (df["Close"].shift(1) <= vw.shift(1)) & (df["Close"] > vw)
    entry = _volume_filter(df, entry, vol_mult, vol_window)
    entry = _adx_filter(df, entry, adx_min)
    exit_ = (df["Close"].shift(1) >= vw.shift(1)) & (df["Close"] < vw)
    desc = (f"Buy when the close crosses above the rolling {vwap_window}-day "
            f"VWAP (daily-bar proxy for session VWAP) on volume above "
            f"{vol_mult:g}x the prior {vol_window}-day average. Exit when "
            f"the close crosses back below VWAP, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def bollinger_rsi(df: pd.DataFrame, bb_period: int = 20,
                  bb_std: float = 2.0, oversold: float = 30,
                  exit_level: float = 50,
                  adx_min: float | None = None) -> dict:
    """Bollinger + RSI mean-reversion: buy only when BOTH say oversold —
    close below the lower band AND RSI below oversold. Exit at the middle
    band or on RSI recovery."""
    mid, upper, lower = bollinger(df["Close"], bb_period, bb_std)
    r = rsi(df["Close"])
    entry = (df["Close"] < lower) & (r < oversold)
    entry = _adx_filter(df, entry, adx_min)
    exit_ = (df["Close"] > mid) | ((r.shift(1) < exit_level)
                                   & (r >= exit_level))
    desc = (f"Buy when the close is below the lower Bollinger band "
            f"({bb_period}, {bb_std:g} std) AND RSI is below {oversold:g}. "
            f"Exit at the middle band or when RSI crosses above "
            f"{exit_level:g}, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def bollinger_breakout(df: pd.DataFrame, bb_period: int = 20,
                       bb_std: float = 2.0,
                       adx_min: float | None = None) -> dict:
    """Bollinger breakout: buy when the close breaks above the upper band.
    Exit when the close falls back below the middle band. The ATR half of
    this combo is the stop — set Stop type to 'ATR multiple' in the
    Filters & stops section."""
    mid, upper, lower = bollinger(df["Close"], bb_period, bb_std)
    entry = (df["Close"].shift(1) <= upper.shift(1)) & (df["Close"] > upper)
    entry = _adx_filter(df, entry, adx_min)
    exit_ = df["Close"] < mid
    desc = (f"Buy when the close breaks above the upper Bollinger band "
            f"({bb_period}, {bb_std:g} std). Exit when the close falls "
            f"below the middle band, or on stop. Pair with the ATR stop "
            f"(Filters & stops) for the Bollinger + ATR combo.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def keltner_breakout(df: pd.DataFrame, ema_period: int = 20,
                     atr_mult: float = 2.0, atr_period: int = 14,
                     adx_min: float | None = None) -> dict:
    """Moving averages + ATR (Keltner Channels): bands are EMA +/- ATR
    multiple, so band width follows volatility. Buy when the close breaks
    above the upper band. Exit when the close falls back below the EMA."""
    mid, upper, lower = keltner_bands(df["Close"], df["High"], df["Low"],
                                      ema_period, atr_mult, atr_period)
    entry = (df["Close"].shift(1) <= upper.shift(1)) & (df["Close"] > upper)
    entry = _adx_filter(df, entry, adx_min)
    exit_ = df["Close"] < mid
    desc = (f"Buy when the close breaks above the upper Keltner band "
            f"(EMA {ema_period} +/- {atr_mult:g}x ATR {atr_period}). Exit "
            f"when the close falls below the EMA, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def vwap_bollinger(df: pd.DataFrame, vwap_window: int = 20,
                   bb_period: int = 20, bb_std: float = 2.0,
                   adx_min: float | None = None) -> dict:
    """VWAP + Bollinger: a Bollinger breakout confirmed by sitting above
    VWAP (both trend filters must agree). Exit when the close falls back
    below VWAP."""
    vw = vwap(df["Close"], df["High"], df["Low"], df["Volume"], vwap_window)
    mid, upper, lower = bollinger(df["Close"], bb_period, bb_std)
    entry = ((df["Close"].shift(1) <= upper.shift(1))
             & (df["Close"] > upper) & (df["Close"] > vw))
    entry = _adx_filter(df, entry, adx_min)
    exit_ = df["Close"] < vw
    desc = (f"Buy when the close breaks above the upper Bollinger band "
            f"({bb_period}, {bb_std:g} std) while above the rolling "
            f"{vwap_window}-day VWAP. Exit when the close falls below "
            f"VWAP, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def dma_crossover(df: pd.DataFrame, fast: int = 9, slow: int = 21,
                  displacement: int = 5,
                  adx_min: float | None = None) -> dict:
    """SMA/EMA + DMA (displaced moving average): the slow EMA is shifted
    forward by `displacement` bars. At each bar the DMA value is the EMA
    from `displacement` bars ago — known at the close, so no lookahead.
    Buy when the fast EMA crosses above the DMA; exit on the cross down."""
    fast_e = ema(df["Close"], fast)
    slow_dma = ema(df["Close"], slow).shift(displacement)
    entry = (fast_e.shift(1) <= slow_dma.shift(1)) & (fast_e > slow_dma)
    entry = _adx_filter(df, entry, adx_min)
    exit_ = (fast_e.shift(1) >= slow_dma.shift(1)) & (fast_e < slow_dma)
    desc = (f"Buy when EMA {fast} crosses above the DMA (EMA {slow} "
            f"displaced forward {displacement} bars). Exit on the cross "
            f"down, or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_.fillna(False),
            "max_holding_bars": None, "describe": desc}


def or_breakout(df: pd.DataFrame, or_minutes: int = 30,
                adx_min: float | None = None) -> dict:
    """True intraday opening-range breakout: buy when the close breaks
    above the high of the first `or_minutes` minutes of the session
    (09:30 ET + or_minutes). Exit at the session close (the engine closes
    positions at each session end) or on stop. Intraday only — on daily
    bars the index has no intraday times, so it never fires."""
    mins = df.index.hour * 60 + df.index.minute
    dates = pd.Series(df.index.date, index=df.index)
    in_or = (mins >= 570) & (mins < 570 + or_minutes)
    after_or = mins >= 570 + or_minutes
    # Session OR high, broadcast to every bar of that session. Only bars
    # AFTER the OR completes can enter, so the OR high is fully known
    # at every entry bar — no lookahead.
    or_high = df["High"].where(in_or).groupby(dates).transform("max")
    same_session = dates == dates.shift(1)
    entry = (after_or & same_session
             & (df["Close"].shift(1) <= or_high.shift(1))
             & (df["Close"] > or_high))
    entry = _adx_filter(df, entry, adx_min)
    exit_ = pd.Series(False, index=df.index)
    desc = (f"Intraday opening-range breakout: buy when the close breaks "
            f"above the high of the first {or_minutes} minutes "
            f"(09:30-{_fmt_or_end(or_minutes)} ET). Exit at the session "
            f"close or on stop.")
    if adx_min is not None:
        desc += f" Entries only when ADX >= {adx_min:g}."
    return {"entry": entry.fillna(False), "exit": exit_,
            "max_holding_bars": None, "describe": desc}


def _fmt_or_end(or_minutes: int) -> str:
    total = 9 * 60 + 30 + or_minutes
    return f"{total // 60:02d}:{total % 60:02d}"


STRATEGIES = {
    "EMA 9/21 crossover": ema_crossover,
    "RSI mean-reversion": rsi_reversion,
    "Prior-day high breakout": prior_high_breakout,
    "RSI + MACD": rsi_macd,
    "VWAP + Volume": vwap_volume,
    "EMA + Volume": ema_crossover,
    "Bollinger + RSI": bollinger_rsi,
    "Bollinger + ATR": bollinger_breakout,
    "Moving averages + ATR": keltner_breakout,
    "VWAP + Bollinger": vwap_bollinger,
    "SMA/EMA + DMA": dma_crossover,
    "Opening-range breakout": or_breakout,
}
