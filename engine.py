"""Backtest engine: long-only, one position at a time, no pyramiding.

Execution model (no lookahead):
- A signal on bar i's close fills at bar i+1's OPEN (pending orders).
- Stops fill intrabar at the stop price (first stop check wins).
- A time-based exit (max_holding_bars) fills at the close of the
  entry bar + max_holding_bars - 1.
- An open position at the end is closed at the last bar's close.
- Slippage worsens every fill: entries pay open * (1 + slip), exits
  receive price * (1 - slip). Commission is charged per side.

Position sizing: fixed $ per trade (no compounding, no leverage).

Stops: fixed % (stop_pct below entry) or ATR multiple
(stop = entry - atr_mult * ATR). An ATR stop is fixed at entry from the
PRIOR bar's ATR — the entry bar's ATR needs that bar's close, so using it
would be lookahead. If the ATR is missing / non-positive (warm-up), the
stop falls back to the fixed % stop.
"""
import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, entry: pd.Series, exit_: pd.Series,
                 max_holding_bars: int | None = None,
                 stop_pct: float = 5.0, commission: float = 0.0,
                 slippage_pct: float = 0.05, capital: float = 100_000.0,
                 position_size: float = 10_000.0,
                 atr: pd.Series | None = None,
                 stop_mode: str = "pct", atr_mult: float = 2.0) -> dict:
    n = len(df)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    entry_sig = entry.to_numpy(bool)
    exit_sig = exit_.to_numpy(bool)
    slip = slippage_pct / 100.0
    atr_arr = atr.to_numpy(float) if atr is not None else None

    def stop_price_for(entry_price: float, entry_idx: int) -> float:
        if stop_mode == "atr" and atr_arr is not None and entry_idx > 0:
            prior_atr = atr_arr[entry_idx - 1]
            if np.isfinite(prior_atr) and prior_atr > 0:
                return entry_price - atr_mult * prior_atr
        return entry_price * (1 - stop_pct / 100.0)

    trades = []
    equity = np.full(n, capital, dtype=float)
    realized = 0.0
    position = None          # dict(entry_idx, entry_price, shares)
    pending_entry = False
    pending_exit = False

    def close_trade(exit_idx, exit_price, exit_reason):
        nonlocal realized, position
        entry_price = position["entry_price"]
        shares = position["shares"]
        gross = shares * (exit_price - entry_price)
        net = gross - 2 * commission
        realized += net
        trades.append({
            "entry_date": df.index[position["entry_idx"]].date(),
            "entry_price": round(entry_price, 2),
            "exit_date": df.index[exit_idx].date(),
            "exit_price": round(exit_price, 2),
            "bars_held": exit_idx - position["entry_idx"] + 1,
            "return_pct": round(net / position_size * 100, 2),
            "pnl": round(net, 2),
            "reason": exit_reason,
        })
        position = None

    for i in range(n):
        # 1) Pending exit (signal from bar i-1) fills at this bar's open.
        if pending_exit and position is not None:
            close_trade(i, opens[i] * (1 - slip), "signal")
            pending_exit = False
        # 2) Pending entry (signal from bar i-1) fills at this bar's open.
        if pending_entry and position is None:
            fill = opens[i] * (1 + slip)
            position = {"entry_idx": i, "entry_price": fill,
                        "shares": position_size / fill,
                        "stop_price": stop_price_for(fill, i)}
            pending_entry = False
        # 3) In-position checks on this bar: stop, then time exit.
        if position is not None:
            stop_price = position["stop_price"]
            if lows[i] <= stop_price:
                close_trade(i, stop_price * (1 - slip), "stop")
            elif (max_holding_bars is not None
                    and i == position["entry_idx"] + max_holding_bars - 1):
                close_trade(i, closes[i] * (1 - slip), "time")
        # 4) New signals on this bar's close -> pending for next bar's open.
        if position is None and entry_sig[i] and i + 1 < n:
            pending_entry = True
        if position is not None and exit_sig[i]:
            if i + 1 < n:
                pending_exit = True
            else:  # signal on the last bar: exit at its close
                close_trade(i, closes[i] * (1 - slip), "signal")
        # 5) Equity at this bar's close (realized + unrealized).
        unrealized = 0.0
        if position is not None:
            unrealized = position["shares"] * (closes[i]
                                               - position["entry_price"])
        equity[i] = capital + realized + unrealized

    # Close any open position at the last close.
    if position is not None:
        close_trade(n - 1, closes[n - 1] * (1 - slip), "end")
        equity[n - 1] = capital + realized

    return {"trades": trades,
            "equity": pd.Series(equity, index=df.index, name="equity"),
            "metrics": compute_metrics(trades, equity, capital)}


def compute_metrics(trades: list, equity: np.ndarray,
                    capital: float) -> dict:
    m = {"trades": len(trades), "wins": 0, "losses": 0, "win_rate": 0.0,
         "avg_win": 0.0, "avg_loss": 0.0, "avg_win_pct": 0.0,
         "avg_loss_pct": 0.0, "expectancy": 0.0, "profit_factor": 0.0,
         "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
         "avg_bars_held": 0.0}
    if capital:
        m["total_return_pct"] = round(
            (equity[-1] - capital) / capital * 100, 2) if len(equity) else 0.0
    if len(equity):
        peak = np.maximum.accumulate(equity)
        dd = np.where(peak > 0, (peak - equity) / peak * 100, 0.0)
        m["max_drawdown_pct"] = round(float(np.max(dd)), 2)
    if not trades:
        return m
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    rets = np.array([t["return_pct"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    m["wins"] = int(len(wins))
    m["losses"] = int(len(losses))
    m["win_rate"] = round(len(wins) / len(pnls) * 100, 1)
    if len(wins):
        m["avg_win"] = round(float(wins.mean()), 2)
        m["avg_win_pct"] = round(float(rets[pnls > 0].mean()), 2)
    if len(losses):
        m["avg_loss"] = round(float(losses.mean()), 2)
        m["avg_loss_pct"] = round(float(rets[pnls <= 0].mean()), 2)
    wr = len(wins) / len(pnls)
    m["expectancy"] = round(wr * m["avg_win"] + (1 - wr) * m["avg_loss"], 2)
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    if gross_loss == 0:
        m["profit_factor"] = float("inf") if gross_profit > 0 else 0.0
    else:
        m["profit_factor"] = round(gross_profit / gross_loss, 2)
    m["avg_bars_held"] = round(
        float(np.mean([t["bars_held"] for t in trades])), 1)
    return m
