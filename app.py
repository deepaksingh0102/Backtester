"""Backtester: run the analyzer's own signals over daily history and measure
what they actually did — win rate, expectancy, profit factor, drawdown."""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data
import engine
import strategies

st.set_page_config(page_title="Backtester", layout="wide")
st.title("📊 Backtester")

with st.sidebar:
    st.header("Data")
    symbol = st.text_input("Symbol", value=config.DEFAULT_SYMBOL).strip().upper()
    today = date.today()
    start = st.date_input("Start", value=today - timedelta(days=365 * config.DEFAULT_YEARS))
    end = st.date_input("End", value=today)
    use_demo = st.toggle("Demo data (offline / no internet)", value=False)

    st.header("Strategy")
    strat_name = st.selectbox("Strategy", list(strategies.STRATEGIES))
    stop_pct = st.number_input("Stop-loss %", min_value=0.0, max_value=50.0,
                               value=float(config.DEFAULT_STOP_PCT), step=0.5)
    if strat_name == "EMA 9/21 crossover":
        fast = st.number_input("Fast EMA", min_value=2, max_value=100, value=9)
        slow = st.number_input("Slow EMA", min_value=3, max_value=200, value=21)
        strat_kwargs = {"fast": int(fast), "slow": int(slow)}
    elif strat_name == "RSI mean-reversion":
        oversold = st.number_input("Oversold RSI", min_value=5.0, max_value=45.0,
                                   value=30.0, step=1.0)
        exit_level = st.number_input("Exit RSI", min_value=40.0, max_value=80.0,
                                     value=50.0, step=1.0)
        strat_kwargs = {"oversold": float(oversold),
                        "exit_level": float(exit_level)}
    else:
        strat_kwargs = {}

    st.header("Costs & sizing")
    commission = st.number_input("Commission $ / side", min_value=0.0,
                                 value=float(config.DEFAULT_COMMISSION),
                                 step=0.5)
    slippage = st.number_input("Slippage % / fill", min_value=0.0,
                               value=float(config.DEFAULT_SLIPPAGE_PCT),
                               step=0.01)
    capital = st.number_input("Starting capital $", min_value=1_000.0,
                              value=float(config.DEFAULT_CAPITAL), step=10_000.0)
    position = st.number_input("Position size $ / trade", min_value=100.0,
                               value=float(config.DEFAULT_POSITION), step=1_000.0)


@st.cache_data(ttl=600, max_entries=16)
def _load(symbol, start, end, use_demo, nonce):
    return data.load_daily(symbol, start, end, use_demo=use_demo)


if not symbol:
    st.info("Type a symbol in the sidebar to begin.")
    st.stop()
if start >= end:
    st.warning("Start date must be before end date.")
    st.stop()

df = _load(symbol, start, end, use_demo, 0)
if df.empty:
    st.error(f"No daily bars for {symbol} in that range.")
    st.stop()

sig = strategies.STRATEGIES[strat_name](df, **strat_kwargs)
res = engine.run_backtest(df, sig["entry"], sig["exit"],
                          max_holding_bars=sig["max_holding_bars"],
                          stop_pct=float(stop_pct),
                          commission=float(commission),
                          slippage_pct=float(slippage),
                          capital=float(capital),
                          position_size=float(position))
m = res["metrics"]
trades = res["trades"]

st.caption(f"{symbol} · {len(df)} daily bars · "
           f"{df.index[0].date()} → {df.index[-1].date()} · "
           f"Source: {'demo' if use_demo else ('Alpaca' if data.alpaca_is_configured() else 'Yahoo')}")
st.subheader(f"{strat_name} — {symbol}")
st.write(sig["describe"])
st.write(f"Execution: signals fill at the NEXT bar's open. Stops fill intrabar "
         f"at the stop price. Stop {stop_pct:g}% · commission "
         f"${commission:g}/side · slippage {slippage:g}%/fill · "
         f"fixed ${position:g} per trade, one position at a time.")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Trades", m["trades"])
c2.metric("Win rate", f"{m['win_rate']}%")
c3.metric("Expectancy / trade", f"${m['expectancy']:,.0f}")
pf = m["profit_factor"]
c4.metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
c5.metric("Max drawdown", f"{m['max_drawdown_pct']}%")
c6.metric("Total return", f"{m['total_return_pct']}%")
st.caption(f"Avg win ${m['avg_win']:,.0f} ({m['avg_win_pct']}%) · "
           f"avg loss ${m['avg_loss']:,.0f} ({m['avg_loss_pct']}%) · "
           f"avg hold {m['avg_bars_held']} bars · "
           f"{m['wins']}W / {m['losses']}L")

# Equity curve
fig = go.Figure()
fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"].values,
                         name="Equity", line=dict(width=2, color="#2563eb")))
fig.add_hline(y=capital, line_dash="dash", line_color="#9ca3af",
              annotation_text="Starting capital",
              annotation_position="bottom right")
fig.update_layout(title="Equity curve", xaxis_title="Date",
                  yaxis_title="Equity ($)", height=380)
st.plotly_chart(fig, use_container_width=True)

if trades:
    st.subheader("Trades")
    st.dataframe(pd.DataFrame(trades), hide_index=True,
                 use_container_width=True)
else:
    st.info("No trades fired in this range for this strategy.")

st.caption("Past performance does not predict future results. A backtest "
           "measures what a fixed rule did on past data — it does not prove "
           "it will work live. Re-tuning parameters until the numbers look "
           "good (overfitting) inflates backtest results. This is a research "
           "tool, not financial advice.")
