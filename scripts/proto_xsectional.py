# scripts/proto_xsectional.py
"""
PROTOTYPE — cross-sectional (relative-value) strategies. The market-neutral vein.

Rank a BASKET of instruments by a signal, then trade the SPREAD between winners and losers:
  - xmom (cross-sectional momentum): long top-N trailing return, short bottom-N
  - xrev (cross-sectional reversion): long bottom-N (laggards), short top-N (leaders)
Dollar-neutral (equal long & short) → no net market exposure → structurally crash-immune
and uncorrelated with every net-long sleeve. A BASKET diversifies the spread (unlike a
single trending pair, e.g. gold/silver, which failed).

Each rebalance: enter long/short legs, hold `hold` days, P&L = mean(long fwd) − mean(short
fwd) − cost. Net of cost (charged on the legs each rebalance).

Usage:
    python scripts/proto_xsectional.py --basket indices --signal xmom --lookback 90 --hold 20
    python scripts/proto_xsectional.py --basket commodities --signal xrev --lookback 5 --hold 5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.utils.logger import get_logger

log = get_logger("proto_xsec")

BASKETS = {
    "indices":     ["usa500idxusd", "usatechidxusd", "usa30idxusd", "deuidxeur", "jpnidxjpy"],
    "commodities": ["XAUUSD", "XAGUSD", "BRENTCMDUSD", "LIGHTCMDUSD", "COPPERCMDUSD", "GASCMDUSD"],
    "metals_energy": ["XAUUSD", "XAGUSD", "BRENTCMDUSD", "LIGHTCMDUSD", "COPPERCMDUSD"],
    # wide FX universe (28 G8 pairs) — breadth for cross-sectional currency momentum
    "fx": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
           "EURGBP", "EURJPY", "EURCHF", "EURAUD", "EURCAD", "EURNZD",
           "GBPJPY", "GBPCHF", "GBPAUD", "GBPCAD", "GBPNZD",
           "AUDJPY", "AUDCHF", "AUDCAD", "AUDNZD",
           "CADJPY", "CADCHF", "CHFJPY", "NZDJPY", "NZDCAD", "NZDCHF"],
}


def to_daily_close(df):
    d = df.resample("1D").agg({"Close": "last"}).dropna()
    return d["Close"]


def load_basket(symbols):
    prov = CSVProvider()
    cols = {}
    for s in symbols:
        try:
            cols[s] = to_daily_close(prov.get_ohlcv(symbol=s.upper(), timeframe="H1"))
        except Exception as e:
            log.warning(f"{s}: skip ({e})")
    df = pd.concat(cols, axis=1, join="inner").dropna()
    return df


def run(closes, signal, lookback, hold, n, cost_pct):
    """closes: DataFrame (dates x symbols), aligned. Returns (date, period_return%) list."""
    C = closes.values
    dates = closes.index
    T, N = C.shape
    n = min(n, N // 2)
    trades = []
    i = lookback
    while i + hold < T:
        trail = C[i] / C[i - lookback] - 1.0          # ranking signal
        order = np.argsort(trail)                      # ascending (loser→winner)
        if signal == "xmom":
            longs, shorts = order[-n:], order[:n]      # long winners, short losers
        else:                                          # xrev
            longs, shorts = order[:n], order[-n:]      # long losers, short winners
        fwd = C[i + hold] / C[i] - 1.0
        pnl = fwd[longs].mean() - fwd[shorts].mean()
        trades.append((dates[i], dates[i + hold], pnl * 100 - cost_pct))   # (entry, exit, ret%)
        i += hold
    return trades


def stats(trades, years):
    r = np.array([t[2] for t in trades])
    if len(r) == 0:
        return None
    w = r[r > 0]; l = r[r < 0]
    pf = w.sum() / -l.sum() if l.sum() else float("inf")
    per_yr = len(r) / years
    sharpe = (r.mean()/r.std()*np.sqrt(per_yr)) if r.std() else 0
    eq = np.cumprod(1 + r/100)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return dict(n=len(r), per_yr=per_yr, win=100*len(w)/len(r), pf=pf,
                sharpe=sharpe, maxdd=dd, tot=r.sum(), eq=eq[-1])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--basket", default="indices", choices=list(BASKETS))
    ap.add_argument("--signal", default="xmom", choices=["xmom", "xrev"])
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--n", type=int, default=2)            # legs per side
    ap.add_argument("--cost-pct", type=float, default=0.25)  # round-trip across legs per rebalance
    a = ap.parse_args(argv)

    closes = load_basket(BASKETS[a.basket])
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    log.info(f"{a.basket}: {closes.shape[1]} instruments, {len(closes)} days, "
             f"{closes.index[0].date()}→{closes.index[-1].date()}")
    s = stats(run(closes, a.signal, a.lookback, a.hold, a.n, a.cost_pct), years)
    if s is None:
        print("NO TRADES"); return
    print("\n" + "="*60)
    print(f"  CROSS-SECTIONAL {a.signal.upper()} — {a.basket} "
          f"(lookback {a.lookback}, hold {a.hold}, n{a.n})")
    print("="*60)
    print(f"  Rebalances:    {s['n']}  ({s['per_yr']:.0f}/yr)")
    print(f"  Win rate:      {s['win']:.1f}%")
    print(f"  Profit factor: {s['pf']:.2f}")
    print(f"  Sharpe:        {s['sharpe']:.2f}")
    print(f"  Max DD:        {s['maxdd']*100:.1f}%")
    print(f"  Total return:  {s['tot']:+.1f}%   (net {a.cost_pct}%/rebalance)")
    print("="*60)


if __name__ == "__main__":
    main(sys.argv[1:])
