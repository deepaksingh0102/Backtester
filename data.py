"""Daily bar loading: Alpaca when keys are set (Streamlit Secrets or env),
else Yahoo Finance. Demo mode returns a deterministic synthetic series."""
import os
from datetime import date, timedelta

import pandas as pd
import requests


def _secrets():
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return {}


def alpaca_is_configured() -> bool:
    sec = _secrets()
    key = os.environ.get("APCA_API_KEY_ID") or sec.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY") or sec.get("APCA_API_SECRET_KEY")
    return bool(key and secret and key != "Dummy" and secret != "Dummy")


def _alpaca_feed() -> str:
    sec = _secrets()
    return (os.environ.get("ALPACA_FEED") or sec.get("ALPACA_FEED")
            or "sip")


def _alpaca_headers():
    sec = _secrets()
    return {
        "APCA-API-KEY-ID": os.environ.get("APCA_API_KEY_ID")
        or sec.get("APCA_API_KEY_ID"),
        "APCA-API-SECRET-KEY": os.environ.get("APCA_API_SECRET_KEY")
        or sec.get("APCA_API_SECRET_KEY"),
    }


def fetch_daily_alpaca(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily bars from Alpaca. sort=desc: Alpaca pages FORWARD from start,
    so a limit smaller than the window returns the OLDEST bars — newest-first
    guarantees this window's bars. Re-sorted ascending below."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start": f"{start.isoformat()}T00:00:00Z",
        "end": f"{end.isoformat()}T23:59:59Z",
        "limit": 10000,
        "feed": _alpaca_feed(),
        "adjustment": "raw",
        "sort": "desc",
    }
    r = requests.get(url, headers=_alpaca_headers(), params=params, timeout=20)
    r.raise_for_status()
    bars = r.json().get("bars") or []
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "time": b["t"],
        "Open": b["o"], "High": b["h"], "Low": b["l"],
        "Close": b["c"], "Volume": b["v"],
    } for b in bars])
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
    return (df.set_index("time").sort_index()
            [["Open", "High", "Low", "Close", "Volume"]])


def fetch_daily_yahoo(symbol: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf
    # yfinance treats `end` as exclusive; +1 day keeps the end date in range.
    df = yf.download(symbol, start=start, end=end + timedelta(days=1),
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                            "Close": "Close", "Volume": "Volume"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()


def demo_daily(symbol: str = "DEMO", bars: int = 500) -> pd.DataFrame:
    """Deterministic synthetic daily series (trend + cycle + noise) for
    offline use. Not real market data."""
    import numpy as np
    rng = np.random.default_rng(42)
    t = np.arange(bars)
    close = 100 + 0.08 * t + 8 * np.sin(t / 23.0) + rng.normal(0, 1.2, bars)
    close = np.maximum(close, 10.0)
    open_ = np.empty(bars)
    open_[0] = close[0] - 0.3
    open_[1:] = close[:-1] + rng.normal(0, 0.4, bars - 1)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.8, bars))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.8, bars))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=bars)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close,
                         "Volume": rng.integers(500_000, 2_000_000, bars)},
                        index=idx)


def load_daily(symbol: str, start: date, end: date,
               use_demo: bool = False) -> pd.DataFrame:
    if use_demo:
        df = demo_daily(symbol)
        return df[(df.index.date >= start) & (df.index.date <= end)]
    if alpaca_is_configured():
        try:
            return fetch_daily_alpaca(symbol, start, end)
        except Exception:
            pass  # fall through to Yahoo on any Alpaca failure
    return fetch_daily_yahoo(symbol, start, end)
