# scripts/robustness_momentum.py
"""
Robustness gauntlet for the momentum edge — the gate between "nice backtest" and
"edge you can fund". Runs three tests on a symbol:

  1. YEAR-BY-YEAR  — is the edge consistent, or one-trend-dependent (e.g. 2020-21)?
  2. OUT-OF-SAMPLE — tune-half vs verify-half: does it hold on unseen data?
  3. PARAM STABILITY — PF across a grid of (donchian, atr-k): a robust edge is a
     plateau, an overfit one is a lonely spike.

Usage:
    python scripts/robustness_momentum.py XAUUSD --cost-pct 0.05
    python scripts/robustness_momentum.py BTCUSD --cost-pct 0.20
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from scripts.proto_momentum import run
from src.utils.logger import get_logger

log = get_logger("robustness")


def stats(trades):
    r = np.array([t[2] for t in trades])
    if len(r) == 0:
        return 0, 0.0, 0.0, 0.0
    w = r[r > 0]; lo = r[r < 0]
    pf = w.sum() / -lo.sum() if lo.sum() else float('inf')
    return len(r), 100*len(w)/len(r), pf, r.sum()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--cost-pct", type=float, default=0.10)
    ap.add_argument("--donchian", type=int, default=480)
    ap.add_argument("--trend", type=int, default=480)
    ap.add_argument("--atr-k", type=float, default=5.0)
    ap.add_argument("--atr-period", type=int, default=14)
    a = ap.parse_args(argv)
    sym = a.symbol.upper()

    df = CSVProvider().get_ohlcv(symbol=sym, timeframe="H1")
    short = True
    base = lambda d: run(d, a.donchian, a.trend, a.atr_period, a.atr_k, a.cost_pct, short)

    print("\n" + "#"*64)
    print(f"  ROBUSTNESS GAUNTLET — {sym}  (donchian {a.donchian}, atr-k {a.atr_k}, "
          f"cost {a.cost_pct}%)")
    print("#"*64)

    # ── 1. YEAR-BY-YEAR ───────────────────────────────────────────────
    trades = base(df)
    s = pd.Series({t[0]: t[2] for t in trades})
    s.index = pd.to_datetime(s.index)
    print("\n[1] YEAR-BY-YEAR")
    print(f"    {'yr':>4} {'trades':>7} {'win%':>6} {'PF':>6} {'sum%':>8}")
    pos_years = 0; tot_years = 0
    for yr, g in s.groupby(s.index.year):
        rr = g.values; w = rr[rr > 0]; l2 = rr[rr < 0]
        pf = w.sum()/-l2.sum() if l2.sum() else float('inf')
        tot_years += 1; pos_years += (rr.sum() > 0)
        print(f"    {yr:>4} {len(rr):>7} {100*len(w)/len(rr):>5.0f}% {pf:>6.2f} {rr.sum():>+8.1f}")
    print(f"    → profitable years: {pos_years}/{tot_years}")

    # ── 2. OUT-OF-SAMPLE (time split) ─────────────────────────────────
    mid = df.index[len(df)//2]
    a_df, b_df = df[df.index < mid], df[df.index >= mid]
    n1, w1, pf1, sum1 = stats(base(a_df))
    n2, w2, pf2, sum2 = stats(base(b_df))
    print("\n[2] OUT-OF-SAMPLE (two halves — same params)")
    print(f"    1st half ({a_df.index[0].date()}–{a_df.index[-1].date()}): "
          f"{n1} tr, win {w1:.0f}%, PF {pf1:.2f}, sum {sum1:+.1f}%")
    print(f"    2nd half ({b_df.index[0].date()}–{b_df.index[-1].date()}): "
          f"{n2} tr, win {w2:.0f}%, PF {pf2:.2f}, sum {sum2:+.1f}%")
    print(f"    → both halves PF>1: {'YES ✅' if pf1>1 and pf2>1 else 'NO ❌'}")

    # ── 3. PARAM STABILITY ────────────────────────────────────────────
    print("\n[3] PARAM STABILITY — PF grid (robust=plateau, overfit=spike)")
    dons = [240, 360, 480, 600, 720]; ks = [3, 4, 5, 6]
    print(f"    {'don\\k':>7}" + "".join(f"{k:>7}" for k in ks))
    plateau = 0; cells = 0
    for d in dons:
        row = f"    {d:>7}"
        for k in ks:
            _, _, pf, _ = stats(run(df, d, a.trend, a.atr_period, k, a.cost_pct, short))
            row += f"{pf:>7.2f}"; cells += 1; plateau += (pf > 1.0)
        print(row)
    print(f"    → cells with PF>1: {plateau}/{cells}  "
          f"({'ROBUST plateau' if plateau >= cells*0.7 else 'fragile' if plateau < cells*0.4 else 'mixed'})")
    print("#"*64)


if __name__ == "__main__":
    main(sys.argv[1:])
