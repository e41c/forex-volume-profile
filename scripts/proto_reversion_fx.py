# scripts/proto_reversion_fx.py
"""
PROTOTYPE — short-term FX mean-reversion (two-sided RSI-2 fade), daily.

A different asset class (currencies) than the equity/metals/energy fleet → a genuinely
uncorrelated edge to lift portfolio Sharpe. FX majors have no structural drift, so this
fades extensions BOTH ways (buy oversold, sell overbought) — pays in chop, risk-on or off.

Logic (daily):
  - Entry: RSI(2) < lo → BUY ; RSI(2) > hi → SELL
  - Exit:  RSI(2) crosses back through 50, or max-hold, or catastrophe stop
  - Optional ranging filter (only fade when not in a strong trend).
  - Net of % cost. One position at a time.

Usage:
    python scripts/proto_reversion_fx.py EURUSD
    python scripts/proto_reversion_fx.py GBPUSD --rsi-lo 5 --rsi-hi 95
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.utils.logger import get_logger

log = get_logger("proto_rev_fx")


def rsi(close, period):
    d = np.diff(close, prepend=close[0])
    gain = np.where(d > 0, d, 0.0); loss = np.where(d < 0, -d, 0.0)
    ag = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = np.divide(ag, al, out=np.full_like(ag, 100.0), where=al > 0)
    return 100 - 100/(1+rs)


def to_daily(df):
    return df.resample("1D").agg({"Open":"first","High":"max","Low":"min",
                                  "Close":"last","Volume":"sum"}).dropna()


def run(df, rsi_period, rsi_lo, rsi_hi, max_hold, stop_pct, cost_pct, trend_flat):
    o = df['Open'].values; h = df['High'].values; l = df['Low'].values; c = df['Close'].values
    n = len(df)
    r = rsi(c, rsi_period)
    sma = pd.Series(c).rolling(100).mean().values   # for optional ranging filter
    times = df.index

    pos = 0; entry = 0.0; stop = 0.0; held = 0; et = None
    trades = []  # (entry_t, exit_t, ret%, risk%)
    start = 105
    for i in range(start, n):
        if pos != 0:
            held += 1
            if pos == 1:
                hit_stop = l[i] <= stop
                done = r[i] > 50 or held >= max_hold or hit_stop
                px = stop if hit_stop else c[i]
                if done:
                    trades.append((et, times[i], (px-entry)/entry*100 - cost_pct, stop_pct)); pos = 0
            else:
                hit_stop = h[i] >= stop
                done = r[i] < 50 or held >= max_hold or hit_stop
                px = stop if hit_stop else c[i]
                if done:
                    trades.append((et, times[i], (entry-px)/entry*100 - cost_pct, stop_pct)); pos = 0
        else:
            # optional ranging filter: price near its 100d mean (within trend_flat %)
            ranging = True if trend_flat <= 0 else (abs(c[i]-sma[i])/sma[i]*100 <= trend_flat)
            if not ranging:
                continue
            if r[i] < rsi_lo:
                pos = 1; entry = c[i]; stop = entry*(1-stop_pct/100); held = 0; et = times[i]
            elif r[i] > rsi_hi:
                pos = -1; entry = c[i]; stop = entry*(1+stop_pct/100); held = 0; et = times[i]
    return trades


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--rsi-period", type=int, default=2)
    ap.add_argument("--rsi-lo", type=float, default=10.0)
    ap.add_argument("--rsi-hi", type=float, default=90.0)
    ap.add_argument("--max-hold", type=int, default=8)
    ap.add_argument("--stop-pct", type=float, default=2.0)
    ap.add_argument("--cost-pct", type=float, default=0.02)
    ap.add_argument("--trend-flat", type=float, default=0.0)  # 0=off; else only fade within N% of 100d mean
    args = ap.parse_args(argv)

    df = to_daily(CSVProvider().get_ohlcv(symbol=args.symbol.upper(), timeframe="H1"))
    log.info(f"{args.symbol}: {len(df):,} daily bars {df.index[0].date()}→{df.index[-1].date()}")
    trades = run(df, args.rsi_period, args.rsi_lo, args.rsi_hi, args.max_hold,
                 args.stop_pct, args.cost_pct, args.trend_flat)
    r = np.array([t[2] for t in trades])
    if len(r) == 0:
        print("NO TRADES"); return
    w = r[r > 0]; l = r[r < 0]
    pf = w.sum()/-l.sum() if l.sum() else float('inf')
    years = (df.index[-1]-df.index[0]).days/365.25
    print("\n" + "="*58)
    print(f"  FX MEAN-REVERSION (RSI{args.rsi_period} two-sided) — {args.symbol.upper()}")
    print("="*58)
    print(f"  Trades:        {len(r)}   ({len(r)/years:.1f}/yr)")
    print(f"  Win rate:      {100*len(w)/len(r):.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg win:       +{w.mean():.2f}%   Avg loss: {l.mean():.2f}%")
    print(f"  Expectancy:    {r.mean():+.3f}% / trade  (net of {args.cost_pct}%)")
    print(f"  Sum returns:   {r.sum():+.1f}%")
    print("="*58)


if __name__ == "__main__":
    main(sys.argv[1:])
