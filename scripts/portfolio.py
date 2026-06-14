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
from scripts.proto_reversion_idx import run as rev_run, to_daily
from scripts.proto_xsectional import run as xsec_run, load_basket, BASKETS
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
# validated index mean-reversion recruits
IDX_REVERSION = ["USA500IDXUSD", "USA30IDXUSD", "USATECHIDXUSD"]
DON, TREND, ATRP, ATRK = 480, 480, 14, 5


def momentum_trades():
    """(entry, exit, raw_return%, source) per momentum recruit."""
    out = []; prov = CSVProvider()
    for sym, long_only, cost in MOMENTUM:
        df = prov.get_ohlcv(symbol=sym, timeframe="H1")
        recs = mom_run(df, DON, TREND, ATRP, ATRK, cost, not long_only)
        out += [(et, xt, ret, f"mom:{sym}") for et, xt, ret, _ in recs]
        log.info(f"  mom {sym}: {len(recs)} trades")
    return out


def idx_reversion_trades():
    """(entry, exit, raw_return%, source) per index mean-reversion recruit."""
    out = []; prov = CSVProvider()
    for sym in IDX_REVERSION:
        df = to_daily(prov.get_ohlcv(symbol=sym, timeframe="H1"))
        recs = rev_run(df, 2, 10, 200, 5, 10, 8.0, 0.03)
        out += [(et, xt, ret, f"rev:{sym}") for et, xt, ret, _ in recs]
        log.info(f"  rev {sym}: {len(recs)} trades")
    return out


def fx_xsec_trades():
    """Market-neutral cross-sectional FX reversion (28-pair basket) → raw = period return%."""
    closes = load_basket(BASKETS["fx"])
    recs = xsec_run(closes, "xrev", 20, 20, 4, 0.10)   # validated config
    log.info(f"  xsec FX (market-neutral): {len(recs)} rebalances")
    return [(et, xt, ret, "xsec:fx") for et, xt, ret in recs]


def eurusd_trades():
    """EURUSD VP reversion via the production backtester → raw = net pips."""
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
    log.info(f"  rev EURUSD: {len(r.trades)} trades")
    return [(t.entry_time, t.exit_time, t.pips_net, "rev:EURUSD") for t in r.trades]


def vol_parity(trades):
    """Scale each SOURCE to unit return-std (volatility parity) → R per trade.
    Fairly mixes pips / momentum% / reversion% so each edge contributes equal risk."""
    by_src = {}
    for tr in trades:
        by_src.setdefault(tr[3], []).append(tr[2])
    scale = {s: (1.0/np.std(v) if np.std(v) > 0 else 0.0) for s, v in by_src.items()}
    return [(et, xt, raw*scale[src], src) for et, xt, raw, src in trades]


def simulate(trades, risk_frac):
    """Shared compounding equity, risk_frac per trade, applied at exit time."""
    trades = sorted(trades, key=lambda x: x[1])
    eq = 1.0; curve = []
    for _, xt, R, _ in trades:
        eq *= (1 + R * risk_frac)
        curve.append((xt, eq))
    return pd.Series({t: e for t, e in curve})


def sleeve_of(src):
    if src.startswith("mom"):     return "momentum"
    if src.startswith("xsec"):    return "xsec"
    if src == "rev:EURUSD":       return "eurusd"
    return "idxrev"


def source_weights(trades):
    """Per-sleeve risk budgeting: within a sleeve, weight each source 1/sqrt(n) so a
    cluster of correlated instruments (the 6 momentum) doesn't dominate combined risk."""
    from collections import defaultdict
    srcs = defaultdict(set)
    for t in trades:
        srcs[sleeve_of(t[3])].add(t[3])
    w = {}
    for sleeve, members in srcs.items():
        for s in members:
            w[s] = 1.0 / np.sqrt(len(members))
    return w


def simulate_rm(trades, risk_frac, weights, dd_thr=0.08, dd_mult=0.5):
    """Risk-managed sim: size each trade at ENTRY from current equity drawdown
    (de-risk when bleeding), apply P&L at exit, compounding. weights = sleeve budget."""
    ev = []
    for i, (et, xt, R, src) in enumerate(trades):
        ev.append((et, 1, i))   # enter (process after exits at same ts)
        ev.append((xt, 0, i))   # exit
    ev.sort(key=lambda e: (e[0], e[1]))
    equity = 1.0; peak = 1.0; sized = {}; curve = []
    for t, kind, i in ev:
        _, _, R, src = trades[i]
        if kind == 1:                                   # enter — size now
            dd = equity/peak - 1
            mult = dd_mult if dd < -dd_thr else 1.0
            sized[i] = R * risk_frac * weights[src] * mult
        else:                                           # exit — book P&L
            equity *= (1 + sized.pop(i, 0.0))
            peak = max(peak, equity)
            curve.append((t, equity))
    return pd.Series({t: e for t, e in curve})


def metrics_from_curve(curve, trades, years):
    final = curve.iloc[-1]
    cagr = final ** (1/years) - 1
    maxdd = (curve / curve.cummax() - 1).min()
    ret = curve.pct_change().dropna()
    sharpe = (ret.mean()/ret.std()*np.sqrt(len(ret)/years)) if ret.std() else 0
    Rs = np.array([t[2] for t in trades]); w = Rs[Rs > 0]; l = Rs[Rs < 0]
    pf = w.sum()/-l.sum() if l.sum() else float('inf')
    return dict(final=final, cagr=cagr, maxdd=maxdd, sharpe=sharpe, pf=pf,
                n=len(Rs), per_yr=len(Rs)/years, win=100*len(w)/len(Rs))


def fundable(trades, target_dd, risk0=0.01):
    """Risk-managed config: sleeve budgeting + de-risk-in-DD + vol-target to target_dd."""
    vp = vol_parity(trades); w = source_weights(trades)
    years = (pd.to_datetime([t[0] for t in trades]).max() -
             pd.to_datetime([t[0] for t in trades]).min()).days / 365.25
    risk = risk0; c = None
    for _ in range(5):
        c = simulate_rm(vp, risk, w, 0.08, 0.5)
        m = metrics_from_curve(c, vp, years)
        if abs(m['maxdd']) > 0.001:
            risk *= target_dd / abs(m['maxdd'])
    return metrics_from_curve(c, vp, years), c


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dd", type=float, default=0.10)
    ap.add_argument("--capital", type=float, default=5000.0)   # account size for $ P&L
    a = ap.parse_args(argv)

    log.info("Building sleeves...")
    base = momentum_trades() + eurusd_trades() + idx_reversion_trades()
    xsec = fx_xsec_trades()

    print("\n" + "="*74)
    print(f"  THE FLEET — does the market-neutral FX sleeve help? (fundable @ {int(a.target_dd*100)}% DD)")
    print("="*74)
    print(f"  {'config':<34}{'/yr':>5}{'PF':>6}{'CAGR':>8}{'maxDD':>8}{'Sharpe':>8}")
    print("  " + "-"*70)
    m_base, _ = fundable(base, a.target_dd)
    print(f"  {'WITHOUT xsec (10 edges)':<34}{m_base['per_yr']:>5.0f}{m_base['pf']:>6.2f}"
          f"{m_base['cagr']*100:>7.1f}%{m_base['maxdd']*100:>7.1f}%{m_base['sharpe']:>8.2f}")
    m_all, c = fundable(base + xsec, a.target_dd)
    print(f"  {'WITH market-neutral FX xsec':<34}{m_all['per_yr']:>5.0f}{m_all['pf']:>6.2f}"
          f"{m_all['cagr']*100:>7.1f}%{m_all['maxdd']*100:>7.1f}%{m_all['sharpe']:>8.2f}")
    print("="*74)
    print(f"  Sharpe {m_base['sharpe']:.2f} → {m_all['sharpe']:.2f}   "
          f"CAGR {m_base['cagr']*100:.1f}% → {m_all['cagr']*100:.1f}%  (both at ~{int(a.target_dd*100)}% DD)")
    print(f"  → on $200k funded @80% split ≈ ${m_all['cagr']*200000*0.8:,.0f}/yr per account")
    print("="*74)

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0f0f1a")
    ax.set_facecolor("#16162a")
    ax.plot(c.index, c.values, color="#22c55e", lw=1.3)
    ax.set_yscale("log")
    ax.set_title(f"THE FLEET — 11 edges, risk-managed  CAGR {m_all['cagr']*100:.1f}%  "
                 f"maxDD {m_all['maxdd']*100:.1f}%  Sharpe {m_all['sharpe']:.2f}", color="#e8e8e0")
    ax.tick_params(colors="#888780"); ax.grid(alpha=0.15)
    out = "data/processed/portfolio_equity.png"
    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#0f0f1a")
    print(f"  equity chart → {out}")

    # ── recent-window P&L on a real account ($) ──────────────────────
    cap = a.capital
    end = c.index[-1]
    print("\n" + "="*74)
    print(f"  RECENT REALISED P&L on ${cap:,.0f} (fundable ~{int(a.target_dd*100)}% DD config)")
    print("="*74)
    for label, days in [("3 months", 91), ("6 months", 182), ("12 months", 365)]:
        seg = c[c.index >= end - pd.Timedelta(days=days)]
        if len(seg) > 1:
            ret = seg.iloc[-1]/seg.iloc[0] - 1
            print(f"  last {label:<10} {ret*100:>+7.2f}%   →   ${cap*ret:>+9,.0f}")
    print("="*74)
    print(f"  ⚠ 3-6 months is a TINY sample (noise, not the edge). At this safe DD the system")
    print(f"    is calibrated to ~{m_all['cagr']*100:.0f}% / yr; its value is over YEARS and SCALED capital.")
    print("="*74)


if __name__ == "__main__":
    main(sys.argv[1:])
