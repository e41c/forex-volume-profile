# scripts/proto_pairs.py
"""
PROTOTYPE — market-neutral pairs / ratio reversion (stat-arb). The crash-protective edge.

Trades the SPREAD between two correlated assets, not their direction. Dollar-neutral
(long one leg, short the other), so it doesn't care if markets crash — it bets the ratio
reverts to its mean. Structurally uncorrelated with the net-long fleet → the drawdown-killer.

Default: gold/silver ratio (XAUUSD/XAGUSD), famously mean-reverting.

Logic (daily):
  ratio = A_close / B_close ; z = (ratio - SMA(z_win)) / STD(z_win)
  - z > +entry  → ratio rich  → SHORT ratio (short A, long B)
  - z < -entry  → ratio cheap → LONG  ratio (long A, short B)
  - exit when |z| < z_exit (reverted) or sign flips or max-hold; spread P&L net of cost.

Usage:
    python scripts/proto_pairs.py XAUUSD XAGUSD
    python scripts/proto_pairs.py XAUUSD XAGUSD --z-entry 2.5 --z-win 120
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.utils.logger import get_logger

log = get_logger("proto_pairs")


def to_daily(df):
    return df.resample("1D").agg({"Open":"first","High":"max","Low":"min",
                                  "Close":"last","Volume":"sum"}).dropna()


def run(a, b, z_win, z_entry, z_exit, max_hold, cost_pct):
    """a,b = aligned daily close arrays. Returns (entry_t, exit_t, ret%, src) records."""
    times = a.index
    A = a.values; B = b.values
    ratio = A / B
    rs = pd.Series(ratio)
    z = ((rs - rs.rolling(z_win).mean()) / rs.rolling(z_win).std()).values

    pos = 0; ai = bi = 0.0; held = 0; et = None
    trades = []
    for i in range(z_win + 1, len(A)):
        if pos != 0:
            held += 1
            # exit when reverted past z_exit, or z crossed through the mean (flip), or time
            flipped = (pos == -1 and z[i] < 0) or (pos == 1 and z[i] > 0)
            if abs(z[i]) < z_exit or flipped or held >= max_hold:
                a_ret = (A[i] - ai) / ai * 100
                b_ret = (B[i] - bi) / bi * 100
                # pos=-1 short-ratio: short A + long B → pnl = b_ret - a_ret
                # pos=+1 long-ratio:  long A + short B → pnl = a_ret - b_ret
                pnl = (b_ret - a_ret) if pos == -1 else (a_ret - b_ret)
                trades.append((et, times[i], pnl - cost_pct, "pairs")); pos = 0
        else:
            if np.isnan(z[i]):
                continue
            if z[i] > z_entry:
                pos = -1; ai = A[i]; bi = B[i]; held = 0; et = times[i]
            elif z[i] < -z_entry:
                pos = 1; ai = A[i]; bi = B[i]; held = 0; et = times[i]
    return trades


def load_pair(sym_a, sym_b):
    prov = CSVProvider()
    a = to_daily(prov.get_ohlcv(symbol=sym_a, timeframe="H1"))["Close"]
    b = to_daily(prov.get_ohlcv(symbol=sym_b, timeframe="H1"))["Close"]
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    j.columns = ["a", "b"]
    return j["a"], j["b"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("sym_a"); ap.add_argument("sym_b")
    ap.add_argument("--z-win", type=int, default=100)
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--max-hold", type=int, default=60)
    ap.add_argument("--cost-pct", type=float, default=0.20)
    a = ap.parse_args(argv)

    ca, cb = load_pair(a.sym_a.upper(), a.sym_b.upper())
    log.info(f"{a.sym_a}/{a.sym_b}: {len(ca):,} aligned daily bars "
             f"{ca.index[0].date()}→{ca.index[-1].date()}")
    trades = run(ca, cb, a.z_win, a.z_entry, a.z_exit, a.max_hold, a.cost_pct)
    r = np.array([t[2] for t in trades])
    if len(r) == 0:
        print("NO TRADES"); return
    w = r[r > 0]; l = r[r < 0]
    pf = w.sum()/-l.sum() if l.sum() else float('inf')
    years = (ca.index[-1]-ca.index[0]).days/365.25
    eq = float(np.prod(1+r/100))
    print("\n" + "="*58)
    print(f"  PAIRS REVERSION (market-neutral) — {a.sym_a.upper()}/{a.sym_b.upper()}")
    print("="*58)
    print(f"  Trades:        {len(r)}   ({len(r)/years:.1f}/yr)")
    print(f"  Win rate:      {100*len(w)/len(r):.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg win:       +{w.mean():.2f}%   Avg loss: {l.mean():.2f}%")
    print(f"  Expectancy:    {r.mean():+.3f}% / trade  (net of {a.cost_pct}%)")
    print(f"  Sum returns:   {r.sum():+.1f}%   |  compounded 1u: {eq:.2f}x")
    print("="*58)


if __name__ == "__main__":
    main(sys.argv[1:])
