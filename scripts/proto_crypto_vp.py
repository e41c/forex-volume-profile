# scripts/proto_crypto_vp.py
"""
PROTOTYPE — Volume-profile swing reversion on CRYPTO (BTC/ETH), H1.

Tests whether our proven EURUSD edge (reversion to volume fair-value in NEUTRAL/ranging
regimes) survives in a LESS-EFFICIENT market. Standalone so the frozen forex code is
untouched. Everything in PERCENT terms (crypto spans 79× in price) and NET of % fees.

Same signal DNA as the forex swing strategy:
  rolling 400-bar H1 volume profile → price near POC/HVN + ADX<25 + NEUTRAL trend +
  volume spike + rejection candle → fade to fair value, TP at nearest LVN, R:R capped 2.0.
Differences for crypto: %-distances (not pips), %-fees (not pip spread), 24/7 (no session).

Usage:
    python scripts/proto_crypto_vp.py BTCUSD
    python scripts/proto_crypto_vp.py ETHUSD --cost-pct 0.15 --poc-zone 0.15
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from src.data.csv_provider import CSVProvider
from src.indicators.volume_profile import build_volume_profile
from src.indicators.trend_filter import calculate_adx, get_trend_state, calculate_atr
from src.utils.logger import get_logger

log = get_logger("proto_crypto")


def run(df, window, poc_zone_pct, min_stop_pct, atr_mult, wick_ratio,
        vol_mult, min_rr, max_rr, cost_pct, adx_max):
    o = df['Open'].values; h = df['High'].values; l = df['Low'].values
    c = df['Close'].values; v = df['Volume'].values.astype(float)
    n = len(df)
    in_trade = False; tr = {}
    rets = []   # net % return per trade

    for i in range(window, n):
        # manage open trade (intrabar)
        if in_trade:
            if tr['dir'] == 'BUY':
                if l[i] <= tr['sl']: _close(rets, tr, tr['sl'], cost_pct); in_trade = False
                elif h[i] >= tr['tp']: _close(rets, tr, tr['tp'], cost_pct); in_trade = False
            else:
                if h[i] >= tr['sl']: _close(rets, tr, tr['sl'], cost_pct); in_trade = False
                elif l[i] <= tr['tp']: _close(rets, tr, tr['tp'], cost_pct); in_trade = False
            continue

        # cheap ADX gate before expensive profile build
        sig = df.iloc[i-300:i+1] if i >= 300 else df.iloc[:i+1]
        if calculate_adx(sig) > adx_max:
            continue

        win = df.iloc[i-window:i]
        price = c[i]
        try:
            lv = build_volume_profile(win, bins=200, pip_size=price * 0.0001)
        except Exception:
            continue

        near_poc = abs(price - lv.poc) / price * 100 <= poc_zone_pct
        non_poc = [cl for cl in lv.hvn_clusters
                   if abs(cl.peak - lv.poc) / price * 100 > poc_zone_pct]
        near_hvn = any(cl.low <= price <= cl.high for cl in non_poc)
        if not (near_poc or near_hvn):
            continue

        if get_trend_state(sig).direction != "NEUTRAL":
            continue

        if i >= 20 and not (v[i] > v[i-20:i].mean() * vol_mult):
            continue

        body = abs(c[i] - o[i])
        up_wick = h[i] - max(c[i], o[i]); dn_wick = min(c[i], o[i]) - l[i]
        bull = dn_wick > body * wick_ratio and c[i] > o[i]
        bear = up_wick > body * wick_ratio and c[i] < o[i]
        if not (bull or bear):
            continue

        atr_pct = calculate_atr(sig) / price * 100
        min_sl = max(atr_pct * atr_mult, min_stop_pct)

        if bull:
            entry = price; sl = l[i] * (1 - 0.0005)
            sl_pct = (entry - sl) / entry * 100
            if sl_pct < min_sl: continue
            lvns = [x for x in lv.lvns if x > entry]
            tp = min(lvns) if lvns else entry * (1 + sl_pct/100 * min_rr)
            rr = ((tp - entry)/entry*100) / sl_pct
            if rr < min_rr: continue
            if rr > max_rr: tp = entry * (1 + sl_pct/100 * max_rr)
            tr = dict(dir='BUY', entry=entry, sl=sl, tp=tp); in_trade = True
        else:
            entry = price; sl = h[i] * (1 + 0.0005)
            sl_pct = (sl - entry) / entry * 100
            if sl_pct < min_sl: continue
            lvns = [x for x in lv.lvns if x < entry]
            tp = max(lvns) if lvns else entry * (1 - sl_pct/100 * min_rr)
            rr = ((entry - tp)/entry*100) / sl_pct
            if rr < min_rr: continue
            if rr > max_rr: tp = entry * (1 - sl_pct/100 * max_rr)
            tr = dict(dir='SELL', entry=entry, sl=sl, tp=tp); in_trade = True

    return rets


def _close(rets, tr, exit_px, cost_pct):
    if tr['dir'] == 'BUY':
        gross = (exit_px - tr['entry']) / tr['entry'] * 100
    else:
        gross = (tr['entry'] - exit_px) / tr['entry'] * 100
    rets.append(gross - cost_pct)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--from", dest="dfrom", default=None)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--poc-zone", type=float, default=0.10)   # % of price
    ap.add_argument("--min-stop", type=float, default=0.30)   # % floor
    ap.add_argument("--atr-mult", type=float, default=0.4)
    ap.add_argument("--wick", type=float, default=1.5)
    ap.add_argument("--vol-mult", type=float, default=1.4)
    ap.add_argument("--min-rr", type=float, default=1.5)
    ap.add_argument("--max-rr", type=float, default=2.0)
    ap.add_argument("--cost-pct", type=float, default=0.20)   # round-trip % cost
    ap.add_argument("--adx", type=float, default=25.0)
    args = ap.parse_args(argv)

    df = CSVProvider().get_ohlcv(symbol=args.symbol.upper(), timeframe="H1")
    if args.dfrom:
        df = df[df.index >= pd.Timestamp(args.dfrom, tz="UTC")]
    log.info(f"{args.symbol}: {len(df):,} H1 bars {df.index[0].date()}→{df.index[-1].date()}")

    rets = np.array(run(df, args.window, args.poc_zone, args.min_stop, args.atr_mult,
                        args.wick, args.vol_mult, args.min_rr, args.max_rr,
                        args.cost_pct, args.adx))
    if len(rets) == 0:
        print("NO TRADES"); return
    wins = rets[rets > 0]; losses = rets[rets < 0]
    pf = wins.sum() / -losses.sum() if losses.sum() else float('inf')
    years = (df.index[-1] - df.index[0]).days / 365.25
    # rough compounding of % returns at fixed fraction (illustrative, not sized)
    equity = float(np.prod(1 + rets/100))
    print("\n" + "=" * 58)
    print(f"  CRYPTO VP SWING — {args.symbol.upper()} H1  (net of {args.cost_pct}% cost)")
    print("=" * 58)
    print(f"  Trades:        {len(rets)}   ({len(rets)/years:.1f}/yr)")
    print(f"  Win rate:      {100*len(wins)/len(rets):.1f}%")
    print(f"  Profit factor: {pf:.2f}")
    print(f"  Avg win:       +{wins.mean():.2f}%   Avg loss: {losses.mean():.2f}%")
    print(f"  Expectancy:    {rets.mean():+.3f}% / trade  (net of cost)")
    print(f"  Sum of returns:{rets.sum():+.1f}%   |  compounded 1u: {equity:.2f}x")
    print("=" * 58)


if __name__ == "__main__":
    main(sys.argv[1:])
