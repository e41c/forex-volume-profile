# scripts/portfolio.py
"""
THE FLEET — portfolio combiner.

Runs all validated edges on ONE shared, compounding equity curve, risk-normalized so
heterogeneous edges combine fairly: every trade risks the same fraction of equity, and
P&L is measured in R-multiples (outcome ÷ initial risk).

  Momentum sleeve (trend-following, Donchian+ATR): DAX, Nikkei, Nasdaq, Brent, Gold, Silver
  Reversion sleeve (VP mean-reversion):            EURUSD

Output: combined CAGR / max-DD / Sharpe / frequency + equity-curve chart. This combined,
low-DD, high-frequency profile is the prop-firm-fundable deliverable.

Usage:
    python scripts/portfolio.py                  # 1% risk/trade
    python scripts/portfolio.py --risk 0.005
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.csv_provider import CSVProvider
from scripts.proto_momentum import run as mom_run
from src.backtester import run_backtest, TradingCosts
from src.config import Config
from src.utils.logger import get_logger

log = get_logger("portfolio")

# validated momentum recruits: (symbol, long_only, round-trip cost %)
MOMENTUM = [
    ("DEUIDXEUR",     True,  0.03),
    ("JPNIDXJPY",     True,  0.03),
    ("USATECHIDXUSD", True,  0.03),
    ("BRENTCMDUSD",   False, 0.08),
    ("XAUUSD",        False, 0.05),
    ("XAGUSD",        False, 0.05),
]
DON, TREND, ATRP, ATRK = 480, 480, 14, 5


def momentum_trades():
    """Return (entry, exit, R, source) for every momentum recruit."""
    out = []
    prov = CSVProvider()
    for sym, long_only, cost in MOMENTUM:
        df = prov.get_ohlcv(symbol=sym, timeframe="H1")
        recs = mom_run(df, DON, TREND, ATRP, ATRK, cost, not long_only)
        for et, xt, ret, risk in recs:
            if risk > 0:
                out.append((et, xt, ret / risk, f"mom:{sym}"))
        log.info(f"  {sym}: {len(recs)} trades")
    return out


def reversion_trades():
    """EURUSD VP reversion via the production backtester → R-multiples."""
    prov = CSVProvider()
    df = prov.get_ohlcv(symbol="EURUSD", timeframe="H1")
    m15 = prov.get_ohlcv(symbol="EURUSD", timeframe="M15")
    m30 = m15.resample("30min").agg({"Open":"first","High":"max","Low":"min",
                                     "Close":"last","Volume":"sum"}).dropna()
    costs = TradingCosts(spread_pips=1.2, slippage_pips=0.5, commission=0.0,
                         swap_long=-0.8, swap_short=0.2)
    r = run_backtest(df=df, df_m15=m30, costs=costs, profile_window=Config.PROFILE_WINDOW,
                     warmup_bars=Config.PROFILE_WINDOW, starting_balance=5000,
                     risk_percent=3.0, use_session_profiles=True,
                     entry_wick_ratio=Config.ENTRY_WICK_RATIO,
                     entry_min_body_pips=Config.ENTRY_MIN_BODY_PIPS, verbose=False)
    out = []
    for t in r.trades:
        sl = abs(t.entry - t.stop_loss) / 0.0001
        if sl > 0:
            out.append((t.entry_time, t.exit_time, t.pips_net / sl, "rev:EURUSD"))
    log.info(f"  EURUSD reversion: {len(r.trades)} trades")
    return out


def simulate(trades, risk_frac):
    """Shared compounding equity, risk_frac of equity per trade, applied at exit time."""
    trades = sorted(trades, key=lambda x: x[1])   # by exit time
    eq = 1.0
    curve = []
    for _, xt, R, _ in trades:
        eq *= (1 + R * risk_frac)
        curve.append((xt, eq))
    return pd.Series({t: e for t, e in curve})


def metrics(curve, trades, risk_frac, years):
    Rs = np.array([t[2] for t in trades])
    final = curve.iloc[-1]
    cagr = final ** (1/years) - 1
    peak = curve.cummax(); dd = (curve/peak - 1); maxdd = dd.min()
    # per-trade return in equity terms, annualized Sharpe
    eqret = Rs * risk_frac
    sharpe = (eqret.mean()/eqret.std()*np.sqrt(len(Rs)/years)) if eqret.std() else 0
    wins = Rs[Rs > 0]; losses = Rs[Rs < 0]
    pf = wins.sum()/-losses.sum() if losses.sum() else float('inf')
    return dict(final=final, cagr=cagr, maxdd=maxdd, sharpe=sharpe, pf=pf,
                n=len(Rs), per_yr=len(Rs)/years, win=100*len(wins)/len(Rs))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk", type=float, default=0.01)   # fraction of equity risked per trade
    a = ap.parse_args(argv)

    log.info("Building momentum sleeve...")
    mom = momentum_trades()
    log.info("Building reversion sleeve (EURUSD backtest ~5min)...")
    rev = reversion_trades()
    allt = mom + rev

    span = pd.to_datetime([t[0] for t in allt])
    years = (span.max() - span.min()).days / 365.25

    curve = simulate(allt, a.risk)
    m = metrics(curve, allt, a.risk, years)
    mom_m = metrics(simulate(mom, a.risk), mom, a.risk, years)
    rev_m = metrics(simulate(rev, a.risk), rev, a.risk, years)

    print("\n" + "="*64)
    print(f"  THE FLEET — combined portfolio  ({a.risk*100:.1f}% risk/trade, {years:.0f}yr)")
    print("="*64)
    print(f"  {'sleeve':<14}{'trades':>7}{'/yr':>6}{'win%':>7}{'PF':>6}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}")
    print("  " + "-"*60)
    for name, mm in (("momentum", mom_m), ("reversion", rev_m), ("COMBINED", m)):
        print(f"  {name:<14}{mm['n']:>7}{mm['per_yr']:>6.0f}{mm['win']:>6.0f}%"
              f"{mm['pf']:>6.2f}{mm['cagr']*100:>7.1f}%{mm['maxdd']*100:>7.1f}%{mm['sharpe']:>8.2f}")
    print("="*64)
    print(f"  Combined: {m['final']:.2f}x over {years:.0f}yr  |  {m['per_yr']:.0f} trades/yr  "
          f"|  DD {m['maxdd']*100:.1f}%")
    print("="*64)

    # equity chart
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0f0f1a")
    ax.set_facecolor("#16162a")
    ax.plot(curve.index, curve.values, color="#22c55e", lw=1.3)
    ax.set_yscale("log")
    ax.set_title(f"THE FLEET — combined equity ({a.risk*100:.1f}% risk/trade)  "
                 f"CAGR {m['cagr']*100:.1f}%  maxDD {m['maxdd']*100:.1f}%  Sharpe {m['sharpe']:.2f}",
                 color="#e8e8e0")
    ax.tick_params(colors="#888780"); ax.grid(alpha=0.15)
    out = "data/processed/portfolio_equity.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"  equity chart → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
