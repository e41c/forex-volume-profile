# scripts/proto_intraday_vp.py
"""
PROTOTYPE — Intraday developing-session volume-profile reversion (EURUSD).

Hypothesis: within a trading day, price reverts to the developing session's volume
fair value (POC). Fade stretches to the value-area edge back toward POC.

  - Reset a developing profile each UTC day; accumulate volume per M5 bar (incremental).
  - After MIN_BARS, when price reaches the developing VAH/VAL with a rejection candle,
    fade it: VAH→SELL / VAL→BUY, target = developing POC, stop = day extreme + buffer.
  - Intraday only: close any open trade at session end (no overnight).
  - Everything measured NET of cost (spread+slippage) — the make-or-break for intraday.

This is a standalone research harness, separate from the production swing backtester.
Self-contained so we can iterate fast and kill it quickly if it doesn't clear costs.

Usage:
    python scripts/proto_intraday_vp.py                 # defaults, full M5 history
    python scripts/proto_intraday_vp.py --from 2015-01-01 --min-target 8 --wick 1.5
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.utils.logger import get_logger

log = get_logger("proto_intraday")

PIP = 0.0001


def value_area(hist: dict, va_pct: float):
    """From {price_bin: volume} return (poc, val, vah) — 70% value area around POC."""
    if not hist:
        return None
    prices = np.array(sorted(hist))
    vols = np.array([hist[p] for p in prices], dtype=float)
    poc_i = int(vols.argmax())
    target = vols.sum() * va_pct
    acc = vols[poc_i]; lo = hi = poc_i
    while acc < target and (lo > 0 or hi < len(prices) - 1):
        up = vols[hi + 1] if hi < len(prices) - 1 else -1
        dn = vols[lo - 1] if lo > 0 else -1
        if up >= dn:
            hi += 1; acc += vols[hi]
        else:
            lo -= 1; acc += vols[lo]
    return prices[poc_i], prices[lo], prices[hi]


def efficiency_ratio(closes: np.ndarray) -> float:
    """Kaufman efficiency ratio over a window: |net move| / total path.
    Near 1 = clean trend; near 0 = choppy/ranging. Reversion wants LOW ER."""
    if len(closes) < 2:
        return 1.0
    net = abs(closes[-1] - closes[0])
    path = np.abs(np.diff(closes)).sum()
    return net / path if path > 0 else 0.0


def run(df, va_pct, min_bars, wick_ratio, min_target_pips, buffer_pips,
        cost_pips, trade_hours, bin_pips, er_win, er_max,
        mode="fade", sl_pips=10.0, tp_pips=15.0):
    """Event loop over M5 bars. Returns list of net-pip results + per-day trade count."""
    o = df['Open'].values; h = df['High'].values; l = df['Low'].values
    c = df['Close'].values; v = df['Volume'].values.astype(float)
    idx = df.index
    dates = idx.normalize()          # UTC day
    hours = idx.hour

    bin_sz = bin_pips * PIP
    hist = {}                         # developing-session histogram
    bars_today = 0
    cur_day = None
    in_trade = False
    tr = {}
    results = []
    day_counts = {}

    h0, h1 = trade_hours

    for i in range(len(df)):
        # ── session reset ────────────────────────────────────────────
        if dates[i] != cur_day:
            if in_trade:             # close at session end (prev close)
                _close(results, tr, c[i-1], cost_pips)
                in_trade = False
            cur_day = dates[i]; hist = {}; bars_today = 0

        # ── accumulate developing profile (distribute vol across H-L) ─
        lo_b = int(round(l[i] / bin_sz)); hi_b = int(round(h[i] / bin_sz))
        n = hi_b - lo_b + 1
        share = v[i] / n
        for b in range(lo_b, hi_b + 1):
            hist[b * bin_sz] = hist.get(b * bin_sz, 0.0) + share
        bars_today += 1

        # ── manage open trade (intrabar SL/TP) ───────────────────────
        if in_trade:
            if tr['dir'] == 'BUY':
                if l[i] <= tr['sl']:
                    _close(results, tr, tr['sl'], cost_pips); in_trade = False
                elif h[i] >= tr['tp']:
                    _close(results, tr, tr['tp'], cost_pips); in_trade = False
            else:
                if h[i] >= tr['sl']:
                    _close(results, tr, tr['sl'], cost_pips); in_trade = False
                elif l[i] <= tr['tp']:
                    _close(results, tr, tr['tp'], cost_pips); in_trade = False
            continue

        # ── entry scan ───────────────────────────────────────────────
        if bars_today < min_bars or not (h0 <= hours[i] < h1):
            continue
        # intraday ranging filter — only fade when price action is choppy (low ER)
        if er_max < 1.0 and i >= er_win:
            if efficiency_ratio(c[i - er_win:i + 1]) > er_max:
                continue
        va = value_area(hist, va_pct)
        if va is None:
            continue
        poc, val, vah = va
        price = c[i]
        body = abs(c[i] - o[i])
        up_wick = h[i] - max(c[i], o[i])
        dn_wick = min(c[i], o[i]) - l[i]

        if mode == "fade":
            # rejection candle at the VA edge → fade back to POC
            bear = up_wick > body * wick_ratio and c[i] < o[i]
            bull = dn_wick > body * wick_ratio and c[i] > o[i]
            if price >= vah and bear and (price - poc) / PIP >= min_target_pips:
                tr = dict(dir='SELL', entry=price, sl=h[i] + buffer_pips * PIP,
                          tp=poc, t=idx[i]); in_trade = True
                day_counts[cur_day] = day_counts.get(cur_day, 0) + 1
            elif price <= val and bull and (poc - price) / PIP >= min_target_pips:
                tr = dict(dir='BUY', entry=price, sl=l[i] - buffer_pips * PIP,
                          tp=poc, t=idx[i]); in_trade = True
                day_counts[cur_day] = day_counts.get(cur_day, 0) + 1
        else:  # mode == "breakout" — go WITH a momentum break of the VA edge
            bull_mom = c[i] > o[i] and body >= up_wick and body >= dn_wick
            bear_mom = c[i] < o[i] and body >= up_wick and body >= dn_wick
            if price >= vah and bull_mom:
                tr = dict(dir='BUY', entry=price, sl=price - sl_pips * PIP,
                          tp=price + tp_pips * PIP, t=idx[i]); in_trade = True
                day_counts[cur_day] = day_counts.get(cur_day, 0) + 1
            elif price <= val and bear_mom:
                tr = dict(dir='SELL', entry=price, sl=price + sl_pips * PIP,
                          tp=price - tp_pips * PIP, t=idx[i]); in_trade = True
                day_counts[cur_day] = day_counts.get(cur_day, 0) + 1

    return results, day_counts


def _close(results, tr, exit_px, cost_pips):
    if tr['dir'] == 'BUY':
        gross = (exit_px - tr['entry']) / PIP
    else:
        gross = (tr['entry'] - exit_px) / PIP
    results.append(gross - cost_pips)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="dfrom", default=None)
    ap.add_argument("--va", type=float, default=0.70)
    ap.add_argument("--min-bars", type=int, default=24)        # ~2h of M5
    ap.add_argument("--wick", type=float, default=1.5)
    ap.add_argument("--min-target", type=float, default=8.0)   # POC must be ≥ N pips away
    ap.add_argument("--buffer", type=float, default=2.0)
    ap.add_argument("--cost", type=float, default=1.7)         # spread+slippage
    ap.add_argument("--hours", default="7,19")                 # UTC trade window
    ap.add_argument("--bin", type=float, default=1.0)          # profile bin size in pips
    ap.add_argument("--er-win", type=int, default=24)          # efficiency-ratio window (M5 bars)
    ap.add_argument("--er-max", type=float, default=1.0)       # only trade if ER <= this (1.0=off)
    ap.add_argument("--mode", choices=["fade", "breakout"], default="fade")
    ap.add_argument("--sl", type=float, default=10.0)          # breakout stop (pips)
    ap.add_argument("--tp", type=float, default=15.0)          # breakout target (pips)
    args = ap.parse_args(argv)

    df = CSVProvider().get_ohlcv(symbol="EURUSD", timeframe="M5")
    if args.dfrom:
        df = df[df.index >= pd.Timestamp(args.dfrom, tz="UTC")]
    h0, h1 = (int(x) for x in args.hours.split(","))
    log.info(f"Running intraday VP reversion on {len(df):,} M5 bars "
             f"{df.index[0].date()} → {df.index[-1].date()}")

    res, day_counts = run(df, args.va, args.min_bars, args.wick, args.min_target,
                          args.buffer, args.cost, (h0, h1), args.bin,
                          args.er_win, args.er_max, args.mode, args.sl, args.tp)

    r = np.array(res)
    n = len(r)
    if n == 0:
        print("NO TRADES — loosen filters"); return
    wins = r[r > 0]; losses = r[r < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() != 0 else float('inf')
    years = (df.index[-1] - df.index[0]).days / 365.25
    print("\n" + "=" * 60)
    print(f"  INTRADAY DEVELOPING-VP REVERSION — EURUSD M5 (net of {args.cost}p cost)")
    print("=" * 60)
    print(f"  Trades:        {n:,}   ({n/years:.0f}/yr, {n/max(len(day_counts),1):.2f}/active-day)")
    print(f"  Win rate:      {100*len(wins)/n:.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Net pips:      {r.sum():,.0f}   (avg {r.mean():+.2f}/trade)")
    print(f"  Avg win:       +{wins.mean():.1f}p   Avg loss: {losses.mean():.1f}p")
    print(f"  Expectancy:    {r.mean():+.2f} pips/trade  (must be > 0 net of cost)")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
