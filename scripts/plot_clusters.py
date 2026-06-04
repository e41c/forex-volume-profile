# scripts/plot_clusters.py
"""
Multi-Timeframe Volume Profile — Presentation Chart

Three-panel layout designed for funding / prop-firm presentations:

  LEFT    (price chart):   Candlesticks + 4-timeframe VP overlay + trade markers
  CENTRE  (VP histogram):  Clean distribution shape — D/P/b profiles visible
  RIGHT   (stats panel):   Level labels + key performance metrics
  BOTTOM  (equity curve):  Cumulative P&L with drawdown shading + trade dots

Usage:
    python scripts/plot_clusters.py              # last 200 H1 bars
    python scripts/plot_clusters.py 300          # last 300 bars
    python scripts/plot_clusters.py 300 2023-01  # 300 bars from 2023-01
    python scripts/plot_clusters.py 200 -- no_trades   # hide trade markers
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive — saves to file without display issues
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from src.data import get_provider
from src.indicators.volume_profile import build_volume_profile
from src.indicators.session_profile import build_session_profile
from src.config import Config

# ─────────────────────────── colour palette ───────────────────────────────────
BG       = '#090912'
PANEL    = '#0f0f20'
PANEL_VP = '#0b0b1c'
PANEL_EQ = '#0d0d1e'
GREEN    = '#22c55e'
RED      = '#f43f5e'
TEXT     = '#e8e8e0'
DIM      = '#9ca3af'
GRID     = '#1a1a30'
WHITE    = '#f1f5f9'

PROFILES = {
    'long_term': dict(color='#3b82f6', alpha=0.18, poc_lw=1.8, label='Long-term (500 bars)'),
    'monthly':   dict(color='#a855f7', alpha=0.22, poc_lw=1.4, label='Monthly  (~22 days)'),
    'weekly':    dict(color='#f97316', alpha=0.30, poc_lw=1.4, label='Weekly   (~5 days)'),
    'daily':     dict(color='#fbbf24', alpha=0.48, poc_lw=2.2, label='Daily    (~24 bars)'),
}

HVN = ['#38bdf8','#818cf8','#34d399','#fb7185',
       '#fbbf24','#a78bfa','#4ade80','#f472b6',
       '#22d3ee','#c084fc','#86efac','#fca5a5']


# ─────────────────────────── helpers ──────────────────────────────────────────

def draw_candles(ax, df):
    n      = len(df)
    body_w = max(0.25, min(0.72, 60 / n))
    for i, (_, r) in enumerate(df.iterrows()):
        up = r['Close'] >= r['Open']
        c  = GREEN if up else RED
        ax.plot([i, i], [r['Low'], r['High']], color=c, lw=0.65, alpha=0.7, zorder=2)
        lo = min(r['Open'], r['Close'])
        h  = max(abs(r['Close'] - r['Open']), r['Close'] * 0.000022)
        ax.add_patch(plt.Rectangle(
            (i - body_w / 2, lo), body_w, h,
            fc=c, ec='none', alpha=0.88, zorder=3
        ))


def overlay_trade_markers(ax, df, trades_df):
    """
    Draw entry arrows and exit marks for trades that fall in the visible range.
    Entry: filled triangle (▲ BUY green / ▼ SELL red)
    Exit:  horizontal tick mark, coloured by outcome (WIN green / LOSS red)
    """
    if trades_df is None or trades_df.empty:
        return

    # Build timestamp → x-index map
    ts_to_x = {ts: i for i, ts in enumerate(df.index)}

    for _, t in trades_df.iterrows():
        entry_ts = t['entry_time']
        exit_ts  = t['exit_time']

        # find closest H1 bar
        entry_x = None
        for ts in df.index:
            if ts >= entry_ts:
                entry_x = ts_to_x[ts]
                break
        if entry_x is None:
            continue

        exit_x = None
        for ts in reversed(df.index.tolist()):
            if ts <= exit_ts:
                exit_x = ts_to_x[ts]
                break

        is_buy  = t['direction'] == 'BUY'
        is_win  = t['result']    == 'WIN'
        e_col   = GREEN if is_buy else RED
        r_col   = GREEN if is_win else RED

        entry_y = t['entry']
        exit_y  = t['exit_price']

        # entry arrow
        marker = '^' if is_buy else 'v'
        offset = -0.0004 if is_buy else +0.0004
        ax.scatter(entry_x, entry_y + offset,
                   marker=marker, color=e_col, s=55, zorder=6,
                   edgecolors='white', linewidths=0.5)

        # trade line (entry → exit)
        if exit_x is not None:
            ax.plot([entry_x, exit_x], [entry_y, exit_y],
                    color=r_col, lw=0.9, alpha=0.55, zorder=4,
                    linestyle='--')

        # exit mark
        if exit_x is not None:
            ax.scatter(exit_x, exit_y, marker='|',
                       color=r_col, s=60, zorder=6,
                       linewidths=1.2, alpha=0.85)


def draw_vp_histogram(ax, profiles, price_lo, price_hi):
    """
    Horizontal histogram bars stacked in the VP panel.
    Longest → shortest so daily profile is on top.
    """
    for key in ['long_term', 'monthly', 'weekly', 'daily']:
        lv  = profiles[key]
        st  = PROFILES[key]

        mask   = (lv.profile.index >= price_lo) & (lv.profile.index <= price_hi)
        s      = lv.profile[mask]
        if s.empty or s.max() == 0:
            continue

        prices = s.index.values
        widths = s.values / s.max()
        bar_h  = (prices[1] - prices[0]) * 1.1 if len(prices) > 1 else 0.0001

        for p, w in zip(prices, widths):
            if w < 0.008:
                continue
            ax.add_patch(plt.Rectangle(
                (0, p - bar_h / 2), w, bar_h,
                fc=st['color'], ec='none', alpha=st['alpha'], zorder=2
            ))

        # POC line
        ax.axhline(lv.poc, color=st['color'], lw=st['poc_lw'],
                   alpha=0.95, zorder=4)
        ax.axhline(lv.vah, color=st['color'], lw=0.5,
                   alpha=0.45, zorder=3, ls=':')
        ax.axhline(lv.val, color=st['color'], lw=0.5,
                   alpha=0.45, zorder=3, ls=':')


def stagger_labels(level_price_dict, min_gap):
    """
    Shift label y-positions upward so no two labels are closer than min_gap.
    Each item in placed is (adjusted_y, key) — y is float, key is str.
    """
    items  = sorted(level_price_dict.items(), key=lambda x: float(x[1]))
    placed = []   # list of (adjusted_y: float, key: str)
    for key, price in items:
        y = float(price)
        # push up past any already-placed label that's too close
        changed = True
        while changed:
            changed = False
            for prev_y, _ in placed:
                if abs(y - prev_y) < min_gap:
                    y = prev_y + min_gap
                    changed = True
        placed.append((y, key))
    return {key: y for y, key in placed}


def draw_right_panel(ax, profiles, lt_clusters, price_lo, price_hi,
                     stats_dict=None, pip_size=0.0001):
    """
    Right panel: level labels (top 55%) + performance stats box (bottom 45%).
    Levels use data (price) coordinates so they align with the shared y-axis.
    Stats box uses axes-fraction coordinates so it always fills its zone cleanly.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(price_lo, price_hi)
    ax.axis('off')

    visible_h  = price_hi - price_lo
    # Reserve bottom 45% of the panel for stats — only draw levels in top 55%
    levels_lo  = price_lo + visible_h * 0.45
    min_gap    = visible_h * 0.015

    # ── Level labels (data coords, shared y-axis) ─────────────────────
    poc_prices = {}
    names      = {'long_term': 'LT', 'monthly': 'Mo', 'weekly': 'Wk', 'daily': 'Day'}
    for key, lv in profiles.items():
        if levels_lo < lv.poc < price_hi:
            poc_prices[key] = lv.poc

    for key, y in stagger_labels(poc_prices, min_gap).items():
        raw = profiles[key].poc
        col = PROFILES[key]['color']
        ax.plot([0, 0.10], [raw, raw], color=col,
                lw=PROFILES[key]['poc_lw'] * 0.65, alpha=0.55)
        ax.text(0.13, y,
                f"{names[key]} POC  {raw:.4f}",
                color=col, fontsize=8.0, va='center', ha='left',
                fontfamily='monospace', fontweight='semibold')

    hvn_prices = {}
    for i, c in enumerate(lt_clusters):
        if levels_lo < c.peak < price_hi:
            hvn_prices[f'_{i}'] = c.peak

    for key, y in stagger_labels(hvn_prices, min_gap).items():
        i   = int(key[1:])
        c   = lt_clusters[i]
        col = HVN[i % len(HVN)]
        w_p = (c.high - c.low) / pip_size
        ax.text(0.13, y,
                f"HVN {i+1}  {c.peak:.4f}  ({w_p:.0f}p)",
                color=col, fontsize=7.5, va='center', ha='left',
                fontfamily='monospace')

    # ── Stats box (axes fraction coords — always bottom 43% of panel) ──
    if stats_dict:
        n_rows   = len(stats_dict)
        box_bot  = 0.01    # axes fraction
        box_top  = 0.42
        row_h    = (box_top - box_bot - 0.06) / max(n_rows, 1)

        # background box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, box_bot), 0.97, box_top - box_bot,
            boxstyle='round,pad=0.005',
            fc='#0e0e26', ec='#2d2d52', lw=0.9,
            transform=ax.transAxes, zorder=1
        ))

        # title
        ax.text(0.50, box_top - 0.01, "BACKTEST  METRICS",
                transform=ax.transAxes,
                color='#4a4a80', fontsize=7.5, fontfamily='monospace',
                fontweight='bold', ha='center', va='top')

        # divider
        ax.axhline(price_lo + visible_h * (box_top - 0.04),
                   color='#2d2d52', lw=0.6, alpha=0.8)

        # rows
        for idx, (label, value, color) in enumerate(stats_dict):
            y_frac = box_top - 0.06 - idx * row_h - row_h * 0.5
            ax.text(0.04, y_frac, label,
                    transform=ax.transAxes,
                    color=DIM, fontsize=7.8, fontfamily='monospace',
                    va='center', ha='left')
            ax.text(0.97, y_frac, value,
                    transform=ax.transAxes,
                    color=color, fontsize=7.8, fontfamily='monospace',
                    va='center', ha='right', fontweight='bold')


def draw_equity_curve(ax, trades_df, start_balance=5000.0):
    """
    Cumulative P&L curve with drawdown shading and per-trade dots.
    """
    if trades_df is None or trades_df.empty:
        ax.text(0.5, 0.5, 'No trades in range', transform=ax.transAxes,
                color=DIM, ha='center', va='center', fontsize=9)
        return

    td = trades_df.copy()
    td['entry_time'] = pd.to_datetime(td['entry_time'], utc=True)
    td = td.sort_values('entry_time').reset_index(drop=True)

    equity  = [start_balance]
    dates   = [td['entry_time'].iloc[0]]
    for _, t in td.iterrows():
        equity.append(equity[-1] + t['pnl_cad'])
        dates.append(t['entry_time'])

    eq  = np.array(equity)
    xs  = np.arange(len(eq))

    # Drawdown shading
    running_max = np.maximum.accumulate(eq)
    ax.fill_between(xs, eq, running_max, where=(eq < running_max),
                    color=RED, alpha=0.18, label='Drawdown', zorder=1)

    # Equity line
    ax.plot(xs, eq, color='#60a5fa', lw=1.6, zorder=3, label='Equity')
    ax.axhline(start_balance, color=DIM, lw=0.6, ls='--', alpha=0.5)

    # Trade dots
    for i, (_, t) in enumerate(td.iterrows()):
        c = GREEN if t['result'] == 'WIN' else RED
        ax.scatter(i + 1, equity[i + 1], color=c, s=18, zorder=4, alpha=0.8)

    # Labels
    final = eq[-1]
    ret   = (final - start_balance) / start_balance * 100
    color = GREEN if final >= start_balance else RED
    ax.text(len(xs) - 1, final,
            f"  {final:,.0f} CAD  ({ret:+.1f}%)",
            color=color, fontsize=7.5, va='center', fontfamily='monospace')

    ax.set_ylim(min(eq) * 0.97, max(eq) * 1.03)
    ax.set_xlim(0, len(eq))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_ylabel('Account (CAD)', color=DIM, fontsize=7.5)

    # X-axis: trade numbers with date labels every Nth trade
    step = max(1, len(td) // 10)
    xt   = list(range(0, len(td) + 1, step))
    xl   = [td['entry_time'].iloc[min(i, len(td) - 1)].strftime('%Y-%m') for i in xt]
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=30, ha='right', fontsize=6.5, color=DIM)

    ax.legend(fontsize=7, facecolor='#111128', labelcolor=TEXT,
              framealpha=0.8, loc='upper left')


def compute_stats(trades_df, start_balance=5000.0):
    """Return list of (label, value_str, color) tuples for the stats panel."""
    if trades_df is None or trades_df.empty:
        return []

    td    = trades_df.copy()
    wins  = td[td['result'] == 'WIN']
    losses= td[td['result'] == 'LOSS']
    n     = len(td)

    win_rate  = len(wins) / n * 100 if n else 0
    pf        = (wins['pnl_cad'].sum() / abs(losses['pnl_cad'].sum())
                 if not losses.empty and losses['pnl_cad'].sum() != 0 else 0)
    net_pnl   = td['pnl_cad'].sum()
    net_pct   = net_pnl / start_balance * 100

    avg_win   = wins['pips_net'].mean()   if not wins.empty else 0
    avg_loss  = losses['pips_net'].mean() if not losses.empty else 0
    avg_rr    = td['rr_ratio'].mean()

    # Drawdown
    eq       = start_balance + td['pnl_cad'].cumsum().values
    eq       = np.concatenate([[start_balance], eq])
    peak     = np.maximum.accumulate(eq)
    dd       = (peak - eq) / peak * 100
    max_dd   = dd.max()

    # Sharpe (approx — daily returns from individual trades)
    td2       = td.copy()
    td2['entry_time'] = pd.to_datetime(td2['entry_time'], utc=True)
    td2 = td2.sort_values('entry_time')
    returns   = td2['pnl_cad'].values
    sharpe    = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0

    calmar    = abs(net_pct / max_dd) if max_dd > 0 else 0

    avg_hold  = td['nights_held'].mean() * 24 if 'nights_held' in td.columns else 0

    G = GREEN
    R = RED
    W = WHITE
    D = DIM

    def pf_col(v):  return G if v >= 1.5 else (W if v >= 1.0 else R)
    def wr_col(v):  return G if v >= 50  else (W if v >= 40  else R)
    def dd_col(v):  return G if v <= 10  else (W if v <= 20  else R)
    def sh_col(v):  return G if v >= 1.0 else (W if v >= 0.5 else R)

    return [
        ("Trades",       f"{n}",                          W),
        ("Win rate",     f"{win_rate:.1f}%",              wr_col(win_rate)),
        ("Profit factor",f"{pf:.2f}",                    pf_col(pf)),
        ("Net P&L",      f"{net_pnl:+,.0f} CAD ({net_pct:+.1f}%)",
                         G if net_pnl > 0 else R),
        ("Max drawdown", f"{max_dd:.1f}%",               dd_col(max_dd)),
        ("Sharpe ratio", f"{sharpe:.2f}",                sh_col(sharpe)),
        ("Calmar ratio", f"{calmar:.2f}",                G if calmar > 1 else W),
        ("Avg win",      f"+{avg_win:.1f} pips",         G),
        ("Avg loss",     f"{avg_loss:.1f} pips",         R),
        ("Avg R:R",      f"{avg_rr:.2f}",                W),
        ("Avg hold",     f"{avg_hold:.1f} hrs",          D),
    ]


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    args     = sys.argv[1:]
    n_bars   = int(args[0]) if args and args[0].lstrip('-').isdigit() else 200
    from_str = args[1] if len(args) > 1 and not args[1].startswith('-') else None
    show_trades = 'no_trades' not in args

    provider = get_provider("offline")   # plotting always reads DuckDB
    df_full  = provider.get_ohlcv('EURUSD', 'H1')

    if from_str:
        mask    = df_full.index >= pd.Timestamp(from_str, tz='UTC')
        df_full = df_full[mask]
        df = df_full.head(n_bars).copy()   # START from the given date
    else:
        df = df_full.tail(n_bars).copy()   # default: most recent N bars
    print(f"Plotting {len(df)} bars:  {df.index[0].date()} → {df.index[-1].date()}")

    # ── Build profiles ────────────────────────────────────────────────
    pf = {
        'long_term': build_volume_profile(df, bins=100),
        'monthly':   build_session_profile(df, 'monthly', bins=100),
        'weekly':    build_session_profile(df, 'weekly',  bins=75),
        'daily':     build_session_profile(df, 'daily',   bins=50),
    }
    lt = pf['long_term']
    print(f"LT POC {lt.poc:.5f}  |  {len(lt.hvn_clusters)} HVNs  |  {len(lt.lvns)} LVNs")

    # ── Load trades ───────────────────────────────────────────────────
    journal_path = 'data/processed/trade_journal.csv'
    trades_df    = None
    all_trades   = None
    if show_trades and os.path.exists(journal_path):
        all_trades = pd.read_csv(journal_path)
        all_trades['entry_time'] = pd.to_datetime(all_trades['entry_time'], utc=True)
        all_trades['exit_time']  = pd.to_datetime(all_trades['exit_time'],  utc=True)
        # Trades visible in this chart window
        chart_start = df.index[0]
        chart_end   = df.index[-1]
        trades_df   = all_trades[
            (all_trades['entry_time'] >= chart_start) &
            (all_trades['entry_time'] <= chart_end)
        ].copy()
        print(f"Trades in window: {len(trades_df)}  |  Total in journal: {len(all_trades)}")

    # ── Price range ───────────────────────────────────────────────────
    pad      = (df['High'].max() - df['Low'].min()) * 0.05
    price_lo = df['Low'].min()  - pad
    price_hi = df['High'].max() + pad

    # ── Figure & GridSpec ─────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 14), facecolor=BG)
    gs  = GridSpec(
        2, 3, figure=fig,
        width_ratios=[58, 17, 25],
        height_ratios=[68, 32],
        wspace=0.0, hspace=0.08,
        left=0.04, right=0.99, top=0.92, bottom=0.07
    )
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vp    = fig.add_subplot(gs[0, 1], sharey=ax_price)
    ax_right = fig.add_subplot(gs[0:, 2])          # spans both rows
    ax_eq    = fig.add_subplot(gs[1, 0:2])          # equity curve

    for ax, bg in [(ax_price, PANEL), (ax_vp, PANEL_VP),
                   (ax_right, BG), (ax_eq, PANEL_EQ)]:
        ax.set_facecolor(bg)
        ax.tick_params(colors=DIM, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

    # ── Price chart ───────────────────────────────────────────────────
    draw_candles(ax_price, df)

    # HVN zone shading + peak lines
    for i, c in enumerate(lt.hvn_clusters):
        col = HVN[i % len(HVN)]
        ax_price.axhspan(c.low, c.high, alpha=0.06, color=col, zorder=1)
        ax_price.axhline(c.peak, color=col, lw=0.9, alpha=0.60, ls='-', zorder=2)
        ax_price.axhline(c.low,  color=col, lw=0.35, alpha=0.28, ls='--', zorder=2)
        ax_price.axhline(c.high, color=col, lw=0.35, alpha=0.28, ls='--', zorder=2)

    # LVNs
    for lvn in lt.lvns:
        ax_price.axhline(lvn, color=DIM, lw=0.45, ls=':', alpha=0.30, zorder=1)

    # Session POC lines
    for key, lv in pf.items():
        st = PROFILES[key]
        ax_price.axhline(lv.poc, color=st['color'], lw=st['poc_lw'] * 0.7,
                         alpha=0.70, ls='-', zorder=3)
        ax_price.axhline(lv.vah, color=st['color'], lw=0.5, alpha=0.35, ls=':', zorder=2)
        ax_price.axhline(lv.val, color=st['color'], lw=0.5, alpha=0.35, ls=':', zorder=2)

    # Trade markers
    if show_trades and trades_df is not None:
        overlay_trade_markers(ax_price, df, trades_df)

    # Axes
    step   = max(1, len(df) // 14)
    xticks = list(range(0, len(df), step))
    ax_price.set_xticks(xticks)
    ax_price.set_xticklabels(
        [df.index[i].strftime('%b %d') for i in xticks],
        rotation=35, ha='right', fontsize=7, color=DIM
    )
    ax_price.set_xlim(-1, len(df) + 1)
    ax_price.set_ylim(price_lo, price_hi)
    ax_price.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.4f}'))
    ax_price.set_ylabel('Price (EURUSD)', color=DIM, fontsize=8)
    ax_price.grid(axis='y', color=GRID, lw=0.3, alpha=0.45)

    # Legend
    leg_h = [mpatches.Patch(fc=PROFILES[k]['color'], alpha=0.55,
                            label=PROFILES[k]['label'])
             for k in ['long_term', 'monthly', 'weekly', 'daily']]
    leg_h += [
        mpatches.Patch(fc=HVN[0], alpha=0.40, label='HVN cluster'),
        plt.Line2D([0],[0], color=DIM, ls=':', lw=0.8, label='LVN (TP target)'),
        plt.Line2D([0],[0], color=GREEN, lw=1.2, label='Buy / Win'),
        plt.Line2D([0],[0], color=RED,   lw=1.2, label='Sell / Loss'),
    ]
    ax_price.legend(handles=leg_h, fontsize=6.8, facecolor='#0d0d28',
                    labelcolor=TEXT, framealpha=0.88, loc='upper left',
                    ncol=2, borderpad=0.5)

    # ── VP histogram panel ─────────────────────────────────────────────
    draw_vp_histogram(ax_vp, pf, price_lo, price_hi)
    ax_vp.set_xlim(0, 1.05)
    ax_vp.yaxis.set_visible(False)
    ax_vp.xaxis.set_visible(False)
    ax_vp.set_title('Volume\nProfile', color=DIM, fontsize=7.5, pad=5)
    ax_vp.text(0.5, 0.005, 'Volume →', transform=ax_vp.transAxes,
               color=DIM, fontsize=6.5, ha='center', va='bottom')

    # ── Right panel: levels + stats ────────────────────────────────────
    stats = compute_stats(all_trades)   # stats from full journal
    draw_right_panel(ax_right, pf, lt.hvn_clusters, price_lo, price_hi,
                     stats_dict=stats)

    # Right panel title
    ax_right.text(0.5, 1.0, 'KEY LEVELS & METRICS',
                  transform=ax_right.transAxes,
                  color=DIM, fontsize=8, ha='center', va='top',
                  fontfamily='monospace', fontweight='bold')

    # ── Equity curve ──────────────────────────────────────────────────
    draw_equity_curve(ax_eq, all_trades)
    ax_eq.set_title('Equity Curve  (all trades — 5 000 CAD start)',
                    color=DIM, fontsize=8, pad=5, loc='left')
    ax_eq.grid(axis='y', color=GRID, lw=0.3, alpha=0.45)
    ax_eq.yaxis.set_label_position('left')

    # ── Title bar ─────────────────────────────────────────────────────
    n_t  = len(all_trades) if all_trades is not None else 0
    fig.suptitle(
        f'EURUSD H1  ·  Multi-Timeframe Volume Profile Strategy  ·  '
        f'{df.index[0].strftime("%b %d %Y")} – {df.index[-1].strftime("%b %d %Y")}  ·  '
        f'{n_t} backtest trades  ·  LT POC {lt.poc:.4f}',
        color=TEXT, fontsize=11, fontweight='bold', y=0.975, x=0.52
    )

    # ── Save ──────────────────────────────────────────────────────────
    outpath = f'data/processed/EURUSD_vp_{n_bars}bars.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f'\nChart saved → {outpath}')


if __name__ == '__main__':
    main()
