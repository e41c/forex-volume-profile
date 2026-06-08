# scripts/proto_reversion_idx.py
"""
PROTOTYPE — short-term mean reversion on equity indices (buy the dip in an uptrend).

The uncorrelated complement to the momentum sleeve: momentum buys breakouts/new highs;
this buys OVERSOLD DIPS inside a longer-term uptrend — opposite trigger, opposite regime,
so it pays when trends chop (when momentum bleeds). Classic Connors RSI(2) dip-buy.

Logic (daily bars, long-only — indices drift up):
  - Regime: close > SMA(trend)  (only buy dips inside an uptrend, not falling knives)
  - Entry:  RSI(2) < oversold   (short-term washout)
  - Exit:   close > SMA(exit)  OR  RSI(2) > 50  OR  max-hold days  OR  catastrophe stop
  - Net of % cost. One position at a time.

Usage:
    python scripts/proto_reversion_idx.py usa500idxusd
    python scripts/proto_reversion_idx.py usatechidxusd --rsi-buy 5 --trend 100
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.utils.logger import get_logger

log = get_logger("proto_rev_idx")


def rsi(close, period):
    d = np.diff(close, prepend=close[0])
    gain = np.where(d > 0, d, 0.0); loss = np.where(d < 0, -d, 0.0)
    ag = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = np.divide(ag, al, out=np.full_like(ag, 100.0), where=al > 0)
    return 100 - 100/(1+rs)


def run(df, rsi_period, rsi_buy, trend, exit_sma, max_hold, stop_pct, cost_pct):
    o = df['Open'].values; h = df['High'].values; l = df['Low'].values; c = df['Close'].values
    n = len(df)
    r = rsi(c, rsi_period)
    sma_t = pd.Series(c).rolling(trend).mean().values
    sma_x = pd.Series(c).rolling(exit_sma).mean().values
    times = df.index

    pos = False; entry = 0.0; stop = 0.0; held = 0; et = None
    trades = []   # (entry_t, exit_t, ret%, risk%)
    start = trend + 1
    for i in range(start, n):
        if pos:
            held += 1
            if l[i] <= stop:                                   # catastrophe stop
                trades.append((et, times[i], (stop-entry)/entry*100 - cost_pct,
                               stop_pct)); pos = False
            elif c[i] > sma_x[i] or r[i] > 50 or held >= max_hold:  # reversion done
                trades.append((et, times[i], (c[i]-entry)/entry*100 - cost_pct,
                               stop_pct)); pos = False
        else:
            if not np.isnan(sma_t[i]) and c[i] > sma_t[i] and r[i] < rsi_buy:
                pos = True; entry = c[i]; stop = entry*(1-stop_pct/100); held = 0; et = times[i]
    return trades


def to_daily(df):
    return df.resample("1D").agg({"Open":"first","High":"max","Low":"min",
                                  "Close":"last","Volume":"sum"}).dropna()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--rsi-period", type=int, default=2)
    ap.add_argument("--rsi-buy", type=float, default=10.0)
    ap.add_argument("--trend", type=int, default=200)      # daily SMA uptrend filter
    ap.add_argument("--exit-sma", type=int, default=5)
    ap.add_argument("--max-hold", type=int, default=10)
    ap.add_argument("--stop-pct", type=float, default=8.0) # catastrophe stop
    ap.add_argument("--cost-pct", type=float, default=0.03)
    args = ap.parse_args(argv)

    df = to_daily(CSVProvider().get_ohlcv(symbol=args.symbol.upper(), timeframe="H1"))
    log.info(f"{args.symbol}: {len(df):,} daily bars {df.index[0].date()}→{df.index[-1].date()}")
    trades = run(df, args.rsi_period, args.rsi_buy, args.trend, args.exit_sma,
                 args.max_hold, args.stop_pct, args.cost_pct)
    rets = np.array([t[2] for t in trades])
    if len(rets) == 0:
        print("NO TRADES"); return
    w = rets[rets > 0]; l = rets[rets < 0]
    pf = w.sum()/-l.sum() if l.sum() else float('inf')
    years = (df.index[-1]-df.index[0]).days/365.25
    eq = float(np.prod(1+rets/100))
    print("\n" + "="*58)
    print(f"  INDEX MEAN-REVERSION (RSI{args.rsi_period}<{args.rsi_buy}, dip in uptrend) — "
          f"{args.symbol.upper()}")
    print("="*58)
    print(f"  Trades:        {len(rets)}   ({len(rets)/years:.1f}/yr)")
    print(f"  Win rate:      {100*len(w)/len(rets):.1f}%   (mean-reversion: high win% normal)")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg win:       +{w.mean():.2f}%   Avg loss: {l.mean():.2f}%")
    print(f"  Expectancy:    {rets.mean():+.3f}% / trade  (net of {args.cost_pct}%)")
    print(f"  Sum returns:   {rets.sum():+.1f}%   |  compounded 1u: {eq:.2f}x")
    print("="*58)


if __name__ == "__main__":
    main(sys.argv[1:])
