# scripts/run_backtest.py
"""
Run the full walk-forward backtest and plot the equity curve.

Usage:
    python scripts/run_backtest.py

Costs modelled:
    - Spread      1.2 pips  (standard forex.com EURUSD)
    - Slippage    0.5 pips  (realistic execution)
    - Commission  $0.00     (standard account, no commission)
    - Swap long  -0.8 pips/night
    - Swap short  0.2 pips/night

To test ECN account costs, change TradingCosts() below.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from src.data.csv_provider import CSVProvider
from src.backtester import (
    run_backtest, print_results,
    save_trade_journal, TradingCosts
)
from src.config import Config
from src.utils.logger import get_logger

log = get_logger("backtest")

BG      = '#0f0f1a'
PANEL   = '#16162a'
GREEN   = '#1D9E75'
RED     = '#E24B4A'
BLUE    = '#378ADD'
ORANGE  = '#D85A30'
TEXT    = '#c8c8c4'
SUBTEXT = '#888780'


def plot_equity_curve(result, costs, symbol=Config.SYMBOL):
    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    fig.suptitle(
        f'{symbol}  —  Backtest Results  (after spread + slippage + swap)  |  '
        f'Win Rate: {result.win_rate}%  |  '
        f'Profit Factor: {result.profit_factor}  |  '
        f'Trades: {result.total_trades:,}',
        color=TEXT, fontsize=12, y=0.97
    )

    gs = gridspec.GridSpec(
        3, 2,
        hspace=0.45, wspace=0.3,
        left=0.07, right=0.97,
        top=0.93,   bottom=0.07
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3a')

    # equity curve
    equity = pd.Series(result.equity_curve)
    ax1.plot(equity.index, equity.values,
             color=BLUE, linewidth=1.2, zorder=3)
    ax1.axhline(result.starting_balance,
                color=SUBTEXT, linewidth=0.8,
                linestyle='--', alpha=0.5,
                label=f'Start ${result.starting_balance:.0f}')
    ax1.fill_between(
        equity.index, result.starting_balance, equity.values,
        where=(equity.values >= result.starting_balance),
        color=GREEN, alpha=0.15
    )
    ax1.fill_between(
        equity.index, result.starting_balance, equity.values,
        where=(equity.values < result.starting_balance),
        color=RED, alpha=0.15
    )

    final_color = GREEN if result.net_profit >= 0 else RED
    ax1.annotate(
        f'  ${result.final_balance:.2f}',
        xy=(len(equity) - 1, equity.iloc[-1]),
        fontsize=9, color=final_color, va='center',
        fontweight='bold'
    )

    ax1.set_title(
        f'Equity Curve  |  '
        f'Start: ${result.starting_balance:.0f}  →  '
        f'Final: ${result.final_balance:.2f}  '
        f'({result.net_profit_pct:+.1f}%)  |  '
        f'Fees paid: ${result.total_costs:.2f} CAD',
        color=TEXT, fontsize=9, pad=8
    )
    ax1.set_ylabel('Balance (CAD)', color=SUBTEXT, fontsize=8)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:.0f}')
    )
    ax1.legend(fontsize=7, facecolor='#1a1a2e',
               labelcolor=TEXT, framealpha=0.7)

    # drawdown
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100

    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color=RED, alpha=0.4)
    ax2.plot(drawdown.index, drawdown.values,
             color=RED, linewidth=0.8)
    ax2.axhline(-20, color=ORANGE, linewidth=0.8,
                linestyle='--', alpha=0.6,
                label='20% danger threshold')
    ax2.set_title(
        f'Drawdown  |  '
        f'Max: {result.max_drawdown_pct:.1f}%  '
        f'({"✅ PASS" if result.max_drawdown_pct < 20 else "❌ FAIL"})',
        color=TEXT, fontsize=9, pad=8
    )
    ax2.set_ylabel('Drawdown %', color=SUBTEXT, fontsize=8)
    ax2.legend(fontsize=7, facecolor='#1a1a2e',
               labelcolor=TEXT, framealpha=0.7)

    # win/loss pie
    if result.trades:
        labels = ['Wins', 'Losses']
        sizes  = [result.wins, result.losses]
        colors = [GREEN, RED]
        nz     = [(l,s,c) for l,s,c in zip(labels,sizes,colors) if s > 0]
        if nz:
            labs, szs, cols = zip(*nz)
            wedges, texts, autotexts = ax3.pie(
                szs, labels=labs, colors=cols,
                autopct='%1.0f%%', startangle=90,
                textprops={'color': TEXT, 'fontsize': 8}
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_color(BG)

    ax3.set_title(
        f'Distribution  |  Win rate: {result.win_rate}%',
        color=TEXT, fontsize=9, pad=8
    )

    # net pips per trade
    if result.trades:
        pips   = [t.pips_net for t in result.trades]
        colors = [GREEN if p > 0 else RED for p in pips]
        ax4.bar(range(len(pips)), pips,
                color=colors, alpha=0.7, width=0.8)
        ax4.axhline(0, color=SUBTEXT, linewidth=0.8)

        if result.avg_win_pips != 0:
            ax4.axhline(result.avg_win_pips, color=GREEN,
                        linewidth=1, linestyle='--', alpha=0.7,
                        label=f'Avg win: +{result.avg_win_pips:.0f}')
        if result.avg_loss_pips != 0:
            ax4.axhline(result.avg_loss_pips, color=RED,
                        linewidth=1, linestyle='--', alpha=0.7,
                        label=f'Avg loss: {result.avg_loss_pips:.0f}')
        ax4.legend(fontsize=7, facecolor='#1a1a2e',
                   labelcolor=TEXT, framealpha=0.7)

    ax4.set_title(
        f'Net Pips Per Trade  |  '
        f'Profit factor: {result.profit_factor}  |  '
        f'Avg cost: {result.avg_cost_pips:.1f} pips',
        color=TEXT, fontsize=9, pad=8
    )
    ax4.set_xlabel('Trade #', color=SUBTEXT, fontsize=8)
    ax4.set_ylabel('Pips (net)', color=SUBTEXT, fontsize=8)

    os.makedirs("data/processed", exist_ok=True)
    outpath = f"data/processed/{symbol}_backtest_results.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
    log.info(f"Chart saved → {outpath}")
    plt.show()


if __name__ == "__main__":
    log.info("=" * 55)
    log.info("GOBLIN BACKTESTER STARTING")
    log.info("=" * 55)

    # ── trading costs — edit these to match your account ─────────
    costs = TradingCosts(
        spread_pips   = 1.2,   # forex.com standard EURUSD
        slippage_pips = 0.5,   # realistic execution slippage
        commission    = 0.0,   # $0 standard account (no commission)
        swap_long     = -0.8,  # pips per night long
        swap_short    =  0.2,  # pips per night short
    )

    # load data
    log.info("Loading EURUSD H1 from DuckDB...")
    provider = CSVProvider()
    df = provider.get_ohlcv(
        symbol    = Config.SYMBOL,
        timeframe = Config.TIMEFRAME_PROFILE,
    )
    log.info(
        f"Loaded {len(df):,} bars  "
        f"{df.index[0].date()} → {df.index[-1].date()}"
    )

    # run backtest
    log.info("Running walk-forward backtest...")
    log.info("Goblin patience required — this takes a few minutes 🐲")

    start_time = time.time()

    result = run_backtest(
        df                   = df,
        costs                = costs,
        profile_window       = 500,
        warmup_bars          = 500,
        starting_balance     = Config.ACCOUNT_BALANCE,
        risk_percent         = Config.RISK_PERCENT,
        use_session_profiles = True,
        verbose              = False,  # set True to see every signal
    )

    elapsed = round(time.time() - start_time, 1)
    log.info(f"Backtest complete in {elapsed}s")

    # print results
    print_results(result, costs=costs, symbol=Config.SYMBOL)

    # save journal and plot
    if result.total_trades > 0:
        save_trade_journal(result)
        plot_equity_curve(result, costs, symbol=Config.SYMBOL)
    else:
        log.warning(
            "No trades generated — filters may be too strict.\n"
            "Try in config.py:\n"
            "  POC_ZONE_PIPS  = 10   (currently 3)\n"
            "  HVN_THRESHOLD  = 0.5  (currently 0.7)"
        )
