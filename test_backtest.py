"""Offline checks for the backtester. Engine tests use hand-crafted
entry/exit signals so every expected price and P&L is known in advance."""
import sys

import numpy as np
import pandas as pd

import data
import engine
import strategies

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        print(f"FAIL {name}")


def close_enough(a, b, tol=0.01):
    return abs(a - b) <= tol


def bars(rows):
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=idx)


def sig(flags, idx):
    return pd.Series(flags, index=idx, dtype=bool)


# --- Engine: basic signal entry/exit, no lookahead -------------------------
# Entry signal bar 0 -> fill at open[1] = 101 (NOT close[0] = 100).
# Exit signal bar 3 -> fill at open[4] = 104.
dfA = bars([(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 101, 102),
            (103, 104, 102, 103), (104, 105, 103, 104), (105, 106, 104, 105)])
resA = engine.run_backtest(
    dfA, sig([True, False, False, False, False, False], dfA.index),
    sig([False, False, False, True, False, False], dfA.index),
    stop_pct=50, commission=0.0, slippage_pct=0.0, position_size=10_000)
tA = resA["trades"]
check("engine: one trade", len(tA) == 1)
check("engine: entry at NEXT open (no lookahead)",
      tA and tA[0]["entry_price"] == 101)
check("engine: exit at next open after signal",
      tA and tA[0]["exit_price"] == 104)
check("engine: pnl = shares * (104-101)",
      tA and close_enough(tA[0]["pnl"], 10_000 / 101 * 3))
check("engine: exit reason signal", tA and tA[0]["reason"] == "signal")
check("engine: bars held", tA and tA[0]["bars_held"] == 4)

# --- Engine: slippage + commission hurt the fill ----------------------------
resB = engine.run_backtest(
    dfA, sig([True, False, False, False, False, False], dfA.index),
    sig([False, False, False, True, False, False], dfA.index),
    stop_pct=50, commission=5.0, slippage_pct=0.1, position_size=10_000)
tB = resB["trades"]
check("engine: slippage+commission pnl",
      tB and close_enough(tB[0]["pnl"], 266.46, tol=0.05))
check("engine: costs reduce pnl",
      tB and tA and tB[0]["pnl"] < tA[0]["pnl"])

# --- Engine: stop fills intrabar at the stop price --------------------------
dfC = bars([(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 90, 95),
            (96, 97, 95, 96)])
resC = engine.run_backtest(
    dfC, sig([True, False, False, False], dfC.index),
    sig([False, False, False, False], dfC.index),
    stop_pct=5.0, commission=0.0, slippage_pct=0.0, position_size=10_000)
tC = resC["trades"]
check("engine: stop exit",
      tC and tC[0]["reason"] == "stop" and tC[0]["exit_price"] == 95.95)
check("engine: stop pnl = -5% of position",
      tC and close_enough(tC[0]["pnl"], -500.0))

# --- Engine: time exit (1-day hold) ------------------------------------------
resD = engine.run_backtest(
    dfA, sig([True, False, False, False, False, False], dfA.index),
    sig([False, False, False, False, False, False], dfA.index),
    max_holding_bars=1, stop_pct=50, commission=0.0, slippage_pct=0.0,
    position_size=10_000)
tD = resD["trades"]
check("engine: 1-day hold exits at entry-day close",
      tD and tD[0]["reason"] == "time" and tD[0]["bars_held"] == 1
      and tD[0]["exit_price"] == 101)

# --- Engine: open position closed at the end ---------------------------------
resE = engine.run_backtest(
    dfA, sig([False, False, False, False, True, False], dfA.index),
    sig([False, False, False, False, False, False], dfA.index),
    stop_pct=50, commission=0.0, slippage_pct=0.0, position_size=10_000)
tE = resE["trades"]
check("engine: open position closed at last close",
      tE and tE[0]["reason"] == "end" and tE[0]["entry_price"] == 105
      and tE[0]["exit_price"] == 105)

# --- Engine: no trades -> zero metrics, flat equity --------------------------
resF = engine.run_backtest(
    dfA, sig([False] * 6, dfA.index), sig([False] * 6, dfA.index),
    stop_pct=50, position_size=10_000)
check("engine: no trades",
      resF["metrics"]["trades"] == 0 and resF["metrics"]["win_rate"] == 0.0
      and (resF["equity"] == 100_000).all())

# --- Engine: metrics on a known win + loss -----------------------------------
dfG = bars([(100, 101, 99, 100), (100, 110, 100, 110), (110, 110, 110, 110),
            (110, 110, 110, 110), (100, 101, 99, 100), (100, 100, 95, 95),
            (95, 95, 95, 95), (95, 95, 95, 95)])
resG = engine.run_backtest(
    dfG, sig([True, False, False, False, True, False, False, False], dfG.index),
    sig([False, False, True, False, False, False, True, False], dfG.index),
    stop_pct=50, commission=0.0, slippage_pct=0.0, position_size=10_000)
mG = resG["metrics"]
check("metrics: win rate 50%", mG["win_rate"] == 50.0)
check("metrics: avg win 1000 / avg loss -500",
      mG["avg_win"] == 1000.0 and mG["avg_loss"] == -500.0)
check("metrics: expectancy 250", mG["expectancy"] == 250.0)
check("metrics: profit factor 2.0", mG["profit_factor"] == 2.0)
check("metrics: max drawdown ~0.5%",
      0.4 < mG["max_drawdown_pct"] < 0.6)
check("metrics: total return 0.5%", mG["total_return_pct"] == 0.5)

# --- compute_metrics directly -------------------------------------------------
mH = engine.compute_metrics(
    [{"pnl": 1000.0, "return_pct": 10.0, "bars_held": 3},
     {"pnl": -500.0, "return_pct": -5.0, "bars_held": 2}],
    np.array([100_000.0, 101_000.0, 100_500.0]), 100_000.0)
check("compute_metrics: expectancy/profit factor",
      mH["expectancy"] == 250.0 and mH["profit_factor"] == 2.0
      and mH["avg_bars_held"] == 2.5)

# --- Strategies ---------------------------------------------------------------
flat = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                     "Close": 100.0},
                    index=pd.bdate_range("2024-01-01", periods=30))
check("strategy: flat series, no EMA signals",
      not strategies.ema_crossover(flat)["entry"].any()
      and not strategies.ema_crossover(flat)["exit"].any())
check("strategy: flat series, no RSI signals",
      not strategies.rsi_reversion(flat)["entry"].any())
check("strategy: flat series, no breakout",
      not strategies.prior_high_breakout(flat)["entry"].any())

down_up = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close":
                        list(np.linspace(110, 100, 20))
                        + list(np.linspace(100, 120, 20))},
                       index=pd.bdate_range("2024-01-01", periods=40))
check("strategy: EMA crossover entry fires on fall-then-rise",
      strategies.ema_crossover(down_up)["entry"].any())
up_down = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close":
                        list(np.linspace(100, 120, 20))
                        + list(np.linspace(120, 100, 20))},
                       index=pd.bdate_range("2024-01-01", periods=40))
check("strategy: EMA crossover exit fires on rise-then-fall",
      strategies.ema_crossover(up_down)["exit"].any())

crash = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close":
                      list(100 - 2 * np.arange(20))
                      + list(62 + 3 * np.arange(10))},
                     index=pd.bdate_range("2024-01-01", periods=30))
check("strategy: RSI recovery entry fires after sharp drop",
      strategies.rsi_reversion(crash)["entry"].any())

bo = bars([(100, 100, 100, 100), (100, 100, 100, 100),
           (100, 100, 100, 100), (100, 105, 100, 105)])
bo_sig = strategies.prior_high_breakout(bo)
check("strategy: breakout fires on close above prior high",
      bool(bo_sig["entry"].iloc[3]) and bo_sig["entry"].iloc[:3].sum() == 0)
check("strategy: breakout is a 1-day hold",
      bo_sig["max_holding_bars"] == 1)

# --- ADX ----------------------------------------------------------------------
trend = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close":
                      list(np.linspace(100, 160, 40))},
                     index=pd.bdate_range("2024-01-01", periods=40))
trend["High"] = trend["Close"] + 1.0
trend["Low"] = trend["Close"] - 1.0
check("ADX: strong trend reads above 25",
      strategies.adx(trend["High"], trend["Low"], trend["Close"]).iloc[-1] > 25)
check("ADX: flat series is NaN (no trend to measure)",
      strategies.adx(flat["High"], flat["Low"], flat["Close"]).isna().all())
check("ADX filter: impossible threshold blocks all entries",
      not strategies.ema_crossover(down_up, adx_min=100)["entry"].any())
_f0 = strategies.ema_crossover(down_up, adx_min=0)["entry"]
_fu = strategies.ema_crossover(down_up)["entry"]
check("ADX filter: zero threshold only blocks warm-up bars",
      not _f0.iloc[:26].any() and _f0.iloc[30:].equals(_fu.iloc[30:]))

# --- Volume confirmation (breakout) ------------------------------------------
vol = pd.DataFrame(
    {"Open": [100, 100, 100, 100, 100], "High": [100, 100, 105, 106, 110],
     "Low": [100, 100, 100, 100, 100], "Close": [100, 100, 105, 106, 110],
     "Volume": [1_000_000, 1_000_000, 500_000, 1_000_000, 3_000_000]},
    index=pd.bdate_range("2024-01-01", periods=5))
vol_sig = strategies.prior_high_breakout(vol, vol_mult=1.5, vol_window=2)
check("volume filter: low-volume breakout skipped, high-volume kept",
      not vol_sig["entry"].iloc[2] and not vol_sig["entry"].iloc[3]
      and bool(vol_sig["entry"].iloc[4]))
check("volume filter: off means every breakout fires",
      strategies.prior_high_breakout(vol)["entry"].iloc[2:].all())

# --- ATR + ATR stop ------------------------------------------------------------
const = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                      "Close": 100.0},
                     index=pd.bdate_range("2024-01-01", periods=30))
check("ATR: flat series reads 0 after warm-up",
      strategies.atr(const["High"], const["Low"], const["Close"]).iloc[-1] == 0)
# ATR stop: entry at open[1] = 101, prior-bar ATR = 2.0 -> stop 101 - 2*2 = 97.
# Bar 2 low (90) hits both the ATR stop (97) and the fallback stop (95.95).
dfATR = bars([(100, 101, 99, 100), (101, 102, 100, 101), (102, 103, 90, 100),
              (98, 99, 97, 98)])
atr_series = pd.Series([2.0] * 4, index=dfATR.index)
resATR = engine.run_backtest(
    dfATR, sig([True, False, False, False], dfATR.index),
    sig([False, False, False, False], dfATR.index),
    stop_pct=5.0, commission=0.0, slippage_pct=0.0, position_size=10_000,
    atr=atr_series, stop_mode="atr", atr_mult=2.0)
tATR = resATR["trades"]
check("ATR stop: exits at entry - 2x prior-bar ATR",
      tATR and tATR[0]["reason"] == "stop" and tATR[0]["exit_price"] == 97.0)
# ATR fallback: all-NaN ATR -> fixed % stop (101 * 0.95 = 95.95).
nan_atr = pd.Series([float("nan")] * 4, index=dfATR.index)
resFB = engine.run_backtest(
    dfATR, sig([True, False, False, False], dfATR.index),
    sig([False, False, False, False], dfATR.index),
    stop_pct=5.0, commission=0.0, slippage_pct=0.0, position_size=10_000,
    atr=nan_atr, stop_mode="atr", atr_mult=2.0)
tFB = resFB["trades"]
check("ATR stop: missing ATR falls back to fixed % stop",
      tFB and tFB[0]["reason"] == "stop" and tFB[0]["exit_price"] == 95.95)

# --- Data ---------------------------------------------------------------------
demo = data.demo_daily()
check("data: demo bars OHLC-consistent",
      len(demo) == 500
      and (demo["High"] >= demo[["Open", "Close"]].max(axis=1)).all()
      and (demo["Low"] <= demo[["Open", "Close"]].min(axis=1)).all()
      and not demo.isna().any().any())
from datetime import date as _date, timedelta as _td
_today = _date.today()
# Demo data ends today, so the range must overlap the recent past.
ranged = data.load_daily("DEMO", _today - _td(days=60), _today,
                         use_demo=True)
check("data: demo load respects date range",
      len(ranged) > 0
      and ranged.index[0].date() >= _today - _td(days=60)
      and ranged.index[-1].date() <= _today)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
