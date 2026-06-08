# scripts/proto_momentum.py
"""
PROTOTYPE — Trend-following / momentum (Donchian breakout + ATR chandelier trail).

For TRENDING instruments where reversion fails: crypto (BTC/ETH) and metals (gold/silver).
This is edge-type #2 for the fleet — naturally uncorrelated with EURUSD reversion
(momentum makes money in trends, reversion in chop).

Logic (price-only — no volume needed, so works on any instrument):
  - Entry: close breaks above the N-bar Donchian high → go LONG (below low → SHORT).
  - Trend filter: only long above the SMA(trend), only short below — skip countertrend chop.
  - Exit: ATR "chandelier" trailing stop — long trails at (highest-high-since-entry − K·ATR).
  - Everything in PERCENT, net of round-trip % cost. One position at a time per symbol.

Usage:
    python scripts/proto_momentum.py BTCUSD
    python scripts/proto_momentum.py XAUUSD --donchian 50 --atr-k 3 --cost-pct 0.08
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.utils.logger import get_logger

log = get_logger("proto_momentum")


def atr(h, l, c, period):
    prev = np.roll(c, 1); prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    out = np.full_like(tr, np.nan)
    if len(tr) >= period:
        out[period-1] = tr[:period].mean()
        a = 1.0 / period
        for i in range(period, len(tr)):
            out[i] = a * tr[i] + (1 - a) * out[i-1]
    return out


def run(df, don, trend_len, atr_period, atr_k, cost_pct, allow_short):
    o = df['Open'].values; h = df['High'].values; l = df['Low'].values; c = df['Close'].values
    n = len(df)
    don_hi = pd.Series(h).rolling(don).max().shift(1).values
    don_lo = pd.Series(l).rolling(don).min().shift(1).values
    sma = pd.Series(c).rolling(trend_len).mean().values
    a = atr(h, l, c, atr_period)

    times = df.index
    pos = 0           # 0 flat, 1 long, -1 short
    entry = 0.0; trail = 0.0; ext = 0.0; et = None; init_risk = 0.0
    # records: (entry_time, exit_time, net_return_pct, initial_risk_pct)
    trades = []
    start = max(don, trend_len, atr_period) + 1

    for i in range(start, n):
        if pos == 0:
            if np.isnan(don_hi[i]) or np.isnan(sma[i]) or np.isnan(a[i]):
                continue
            if c[i] > don_hi[i] and c[i] > sma[i]:
                pos = 1; entry = c[i]; ext = h[i]; trail = ext - atr_k * a[i]; et = times[i]
                init_risk = (entry - trail) / entry * 100
            elif allow_short and c[i] < don_lo[i] and c[i] < sma[i]:
                pos = -1; entry = c[i]; ext = l[i]; trail = ext + atr_k * a[i]; et = times[i]
                init_risk = (trail - entry) / entry * 100
        elif pos == 1:
            ext = max(ext, h[i]); trail = max(trail, ext - atr_k * a[i])
            if l[i] <= trail:
                trades.append((et, times[i], (trail - entry)/entry*100 - cost_pct, init_risk)); pos = 0
        else:  # short
            ext = min(ext, l[i]); trail = min(trail, ext + atr_k * a[i])
            if h[i] >= trail:
                trades.append((et, times[i], (entry - trail)/entry*100 - cost_pct, init_risk)); pos = 0
    return trades


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--from", dest="dfrom", default=None)
    ap.add_argument("--donchian", type=int, default=50)     # breakout lookback (H1 bars)
    ap.add_argument("--trend", type=int, default=200)        # SMA trend filter
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-k", type=float, default=3.0)      # chandelier multiple
    ap.add_argument("--cost-pct", type=float, default=0.10)  # round-trip %
    ap.add_argument("--long-only", action="store_true")
    args = ap.parse_args(argv)

    df = CSVProvider().get_ohlcv(symbol=args.symbol.upper(), timeframe="H1")
    if args.dfrom:
        df = df[df.index >= pd.Timestamp(args.dfrom, tz="UTC")]
    log.info(f"{args.symbol}: {len(df):,} H1 bars {df.index[0].date()}→{df.index[-1].date()}")

    trades = run(df, args.donchian, args.trend, args.atr_period, args.atr_k,
                 args.cost_pct, not args.long_only)
    rets = np.array([t[2] for t in trades])
    if len(rets) == 0:
        print("NO TRADES"); return
    wins = rets[rets > 0]; losses = rets[rets < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() else float('inf')
    years = (df.index[-1] - df.index[0]).days / 365.25
    equity = float(np.prod(1 + rets/100))
    print("\n" + "=" * 58)
    print(f"  MOMENTUM (Donchian{args.donchian}+ATR{args.atr_k}x) — {args.symbol.upper()} H1  "
          f"(net {args.cost_pct}%)")
    print("=" * 58)
    print(f"  Trades:        {len(rets)}   ({len(rets)/years:.1f}/yr)")
    print(f"  Win rate:      {100*len(wins)/len(rets):.1f}%   (trend-following: low win% normal)")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg win:       +{wins.mean():.2f}%   Avg loss: {losses.mean():.2f}%")
    print(f"  Expectancy:    {rets.mean():+.3f}% / trade")
    print(f"  Sum returns:   {rets.sum():+.1f}%   |  compounded 1u: {equity:.2f}x")
    print("=" * 58)


if __name__ == "__main__":
    main(sys.argv[1:])
