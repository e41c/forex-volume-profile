# scripts/screen_reversion.py
"""
Gauntlet/screener for the index mean-reversion edge — same rigor as the momentum screen.
OOS (both halves PF>1) + param-plateau (rsi_buy x trend grid) + recent-3yr, ranked.

Usage:
    python scripts/screen_reversion.py usa500idxusd usatechidxusd usa30idxusd jpnidxjpy deuidxeur
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from scripts.proto_reversion_idx import run, to_daily
from src.utils.logger import get_logger

log = get_logger("screen_rev")
COST = 0.03

def pf_sum(trades):
    r = np.array([t[2] for t in trades])
    if len(r) == 0: return 0, 0.0, 0.0
    w = r[r > 0]; l = r[r < 0]
    return len(r), (w.sum()/-l.sum() if l.sum() else 9.99), r.sum()

# fixed exit/stop/hold; vary the entry knobs (rsi_buy, trend)
def cfg(df, rsi_buy, trend):
    return run(df, 2, rsi_buy, trend, 5, 10, 8.0, COST)

def screen(sym):
    try:
        df = to_daily(CSVProvider().get_ohlcv(symbol=sym.upper(), timeframe="H1"))
    except Exception:
        return None
    if len(df) < 800: return None
    n, pf, tot = pf_sum(cfg(df, 10, 200))
    mid = df.index[len(df)//2]
    _, pf1, _ = pf_sum(cfg(df[df.index < mid], 10, 200))
    _, pf2, _ = pf_sum(cfg(df[df.index >= mid], 10, 200))
    oos = pf1 > 1 and pf2 > 1
    green = cells = 0
    for rb in (5, 10, 15):
        for tr in (100, 150, 200):
            _, p, _ = pf_sum(cfg(df, rb, tr)); cells += 1; green += (p > 1)
    cut = df.index[-1] - pd.Timedelta(days=365*3)
    _, _, rec = pf_sum(cfg(df[df.index >= cut], 10, 200))
    yrs = (df.index[-1]-df.index[0]).days/365.25
    return dict(sym=sym.upper(), n=n, pf=pf, tot=tot, peryr=n/yrs,
                pf1=pf1, pf2=pf2, oos=oos, green=green, cells=cells, rec=rec)

def main(argv):
    ap = argparse.ArgumentParser(); ap.add_argument("symbols", nargs="+")
    rows = []
    for s in ap.parse_args(argv).symbols:
        log.info(f"screening {s} ...")
        r = screen(s)
        if r: rows.append(r)
    rows.sort(key=lambda r: (r["oos"], r["green"], r["pf"]), reverse=True)
    print("\n" + "="*88)
    print("  INDEX MEAN-REVERSION SCREEN  (RSI2 dip-buy; OOS = both halves PF>1)")
    print("="*88)
    print(f"  {'instrument':<16}{'PF':>6}{'/yr':>5}{'tot%':>8}{'OOS h1/h2':>13}{'plateau':>9}{'rec3y%':>8}  verdict")
    print("  " + "-"*84)
    for r in rows:
        v = "✅ RECRUIT" if (r["oos"] and r["green"] >= 7 and r["rec"] > 0) else \
            "🟡 maybe" if (r["oos"] and r["green"] >= 6) else "❌ reject"
        print(f"  {r['sym']:<16}{r['pf']:>6.2f}{r['peryr']:>5.0f}{r['tot']:>+8.0f}"
              f"{r['pf1']:>6.2f}/{r['pf2']:<6.2f}{r['green']:>4}/{r['cells']:<4}{r['rec']:>+8.0f}  {v}")
    print("="*88)
    rec = [r['sym'] for r in rows if r['oos'] and r['green'] >= 7 and r['rec'] > 0]
    print(f"  REVERSION RECRUITS: {rec or 'none'}")
    print("="*88)

if __name__ == "__main__":
    main(sys.argv[1:])
