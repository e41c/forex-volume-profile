# scripts/screen_momentum.py
"""
Fleet screener — run the momentum gauntlet's KEY tests across many instruments and rank.

For each symbol: full-period PF, out-of-sample (both halves PF>1?), param-plateau count,
and recent-3-years P&L (is the edge still alive?). One ranked table → spot the soldiers
worth recruiting into the fleet, and the mirages to discard.

Usage:
    python scripts/screen_momentum.py usa500idxusd usatechidxusd lightcmdusd ...
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from scripts.proto_momentum import run
from src.utils.logger import get_logger

log = get_logger("screen")

# conservative round-trip % cost by instrument class
COST = {"idx": 0.03, "cmd": 0.08, "metal": 0.05, "crypto": 0.20, "default": 0.05}

def cost_for(sym):
    s = sym.lower()
    if s in ("btcusd", "ethusd"): return COST["crypto"]   # crypto fees are ~4x forex/metals
    if "idx" in s: return COST["idx"]
    if "cmd" in s: return COST["cmd"]
    if s in ("xauusd", "xagusd"): return COST["metal"]
    return COST["default"]

def long_only_for(sym):
    # Equity indices have structural upward drift — shorting them bleeds. Long-only.
    return "idx" in sym.lower()


def pf_sum(trades):
    r = np.array([t[2] for t in trades])
    if len(r) == 0: return 0, 0.0, 0.0
    w = r[r > 0]; l = r[r < 0]
    return len(r), (w.sum()/-l.sum() if l.sum() else 9.99), r.sum()


def screen(sym):
    try:
        df = CSVProvider().get_ohlcv(symbol=sym.upper(), timeframe="H1")
    except Exception:
        return None
    if len(df) < 5000:
        return None
    cost = cost_for(sym)
    sh = not long_only_for(sym)   # indices long-only

    # full period (donchian 480 / atr-k 5 reference config)
    n, pf, tot = pf_sum(run(df, 480, 480, 14, 5, cost, sh))

    # OOS halves
    mid = df.index[len(df)//2]
    _, pf1, _ = pf_sum(run(df[df.index < mid], 480, 480, 14, 5, cost, sh))
    _, pf2, _ = pf_sum(run(df[df.index >= mid], 480, 480, 14, 5, cost, sh))
    oos = pf1 > 1 and pf2 > 1

    # param plateau (3x3 grid)
    cells = 0; green = 0
    for d in (360, 480, 600):
        for k in (4, 5, 6):
            _, p, _ = pf_sum(run(df, d, 480, 14, k, cost, sh))
            cells += 1; green += (p > 1)

    # recent 3 yrs
    cut = df.index[-1] - pd.Timedelta(days=365*3)
    _, _, rec = pf_sum(run(df[df.index >= cut], 480, 480, 14, 5, cost, sh))

    yrs = (df.index[-1]-df.index[0]).days/365.25
    return dict(sym=sym.upper(), n=n, pf=pf, tot=tot, peryr=n/yrs,
                pf1=pf1, pf2=pf2, oos=oos, plateau=f"{green}/{cells}",
                green=green, rec3=rec, cost=cost)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    a = ap.parse_args(argv)

    rows = []
    for s in a.symbols:
        log.info(f"screening {s} ...")
        r = screen(s)
        if r: rows.append(r)

    # rank: OOS-pass first, then plateau strength, then PF
    rows.sort(key=lambda r: (r["oos"], r["green"], r["pf"]), reverse=True)

    print("\n" + "="*92)
    print("  MOMENTUM FLEET SCREEN  (Donchian480/atr5 ref; OOS = both halves PF>1)")
    print("="*92)
    print(f"  {'instrument':<16}{'PF':>6}{'/yr':>5}{'tot%':>8}{'OOS h1/h2':>12}"
          f"{'plateau':>9}{'rec3y%':>8}  verdict")
    print("  " + "-"*88)
    for r in rows:
        verdict = "✅ RECRUIT" if (r["oos"] and r["green"] >= 7 and r["rec3"] > 0) else \
                  "🟡 maybe" if (r["oos"] and r["green"] >= 6) else "❌ reject"
        print(f"  {r['sym']:<16}{r['pf']:>6.2f}{r['peryr']:>5.0f}{r['tot']:>+8.0f}"
              f"{r['pf1']:>6.2f}/{r['pf2']:<5.2f}{r['plateau']:>9}{r['rec3']:>+8.0f}  {verdict}")
    print("="*92)
    recruits = [r['sym'] for r in rows if r['oos'] and r['green'] >= 7 and r['rec3'] > 0]
    print(f"  FLEET RECRUITS: {recruits or 'none yet'}")
    print("="*92)


if __name__ == "__main__":
    main(sys.argv[1:])
