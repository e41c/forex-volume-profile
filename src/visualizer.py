# src/visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from src.indicators.volume_profile import VolumeProfileLevels
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

BG      = '#0f0f1a'
PANEL   = '#16162a'
BLUE    = '#378ADD'
RED     = '#E24B4A'
GREEN   = '#1D9E75'
ORANGE  = '#D85A30'
MUTED   = '#2a2a4a'
TEXT    = '#c8c8c4'
SUBTEXT = '#888780'


def plot_volume_profile(df: pd.DataFrame,
                        levels: VolumeProfileLevels,
                        symbol: str = Config.SYMBOL,
                        last_n_bars: int = 300):

    plot_df = df.tail(last_n_bars).copy()

    # price window for the visible bars
    p_ymin = plot_df['Low'].min()  * 0.9985
    p_ymax = plot_df['High'].max() * 1.0015

    # profile spans the FULL dataset price range
    prof_ymin = float(levels.profile.index.min())
    prof_ymax = float(levels.profile.index.max())

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    fig.suptitle(
        f'{symbol}  —  Volume Profile Analysis  '
        f'(profile built on {len(df):,} bars · '
        f'{df.index[0].strftime("%b %Y")} → {df.index[-1].strftime("%b %Y")})',
        color=TEXT, fontsize=13, fontweight='normal', y=0.97
    )

    # two independent axes — no sharey
    gs  = gridspec.GridSpec(
        1, 2,
        width_ratios=[3, 1],
        wspace=0.04,
        left=0.06, right=0.97,
        top=0.92,  bottom=0.10
    )
    ax1 = fig.add_subplot(gs[0])   # price chart
    ax2 = fig.add_subplot(gs[1])   # volume profile — independent y axis

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3a')

    # ── PRICE CHART ─────────────────────────────────────────────────
    ax1.plot(plot_df.index, plot_df['Close'],
             color=BLUE, linewidth=1.0, zorder=3)

    # value area shading (only if levels are in visible window)
    if levels.val < p_ymax and levels.vah > p_ymin:
        shade_low  = max(levels.val, p_ymin)
        shade_high = min(levels.vah, p_ymax)
        ax1.axhspan(shade_low, shade_high,
                    alpha=0.08, color=GREEN, zorder=0)

    # draw key levels — if outside visible window add a text annotation instead
    def draw_level(ax, price, color, lw, ls, label):
        if p_ymin <= price <= p_ymax:
            ax.axhline(price, color=color, linewidth=lw,
                       linestyle=ls, alpha=0.9, zorder=4)
            ax.annotate(f' {label}  {price:.5f}',
                        xy=(plot_df.index[0], price),
                        fontsize=7.5, color=color, va='center')
        else:
            direction = '▲' if price > p_ymax else '▼'
            dist_pips = abs(price - plot_df['Close'].iloc[-1]) / 0.0001
            ax.text(0.01, 0.99 if price > p_ymax else 0.01,
                    f'{direction} {label} {price:.5f}  ({dist_pips:.0f} pips away)',
                    transform=ax.transAxes,
                    fontsize=7.5, color=color, va='top' if price > p_ymax else 'bottom',
                    ha='left',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor=PANEL, edgecolor=color,
                              alpha=0.85))

    draw_level(ax1, levels.poc, RED,   1.5, '--', 'POC')
    draw_level(ax1, levels.vah, GREEN, 1.0, '--', 'VAH')
    draw_level(ax1, levels.val, GREEN, 1.0, '--', 'VAL')

    # HVNs visible in price window
    visible_hvns = [h for h in levels.hvns if p_ymin < h < p_ymax]
    for i, hvn in enumerate(visible_hvns):
        ax1.axhline(hvn, color=GREEN, linewidth=0.5,
                    linestyle=':', alpha=0.5, zorder=2)

    # 5 closest LVNs to current price
    last_price   = plot_df['Close'].iloc[-1]
    visible_lvns = sorted(
        [l for l in levels.lvns if p_ymin < l < p_ymax],
        key=lambda x: abs(x - last_price)
    )[:5]
    for lvn in visible_lvns:
        ax1.axhline(lvn, color=ORANGE, linewidth=0.5,
                    linestyle=':', alpha=0.4, zorder=2)

    # current price label
    ax1.annotate(
        f'  {last_price:.5f}',
        xy=(plot_df.index[-1], last_price),
        fontsize=8.5, color=BLUE, va='center', fontweight='bold'
    )

    ax1.set_ylim(p_ymin, p_ymax)
    ax1.set_ylabel('Price', color=SUBTEXT, fontsize=9, labelpad=8)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.4f}')
    )
    ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax1.xaxis.get_majorticklabels(),
             rotation=30, ha='right', color=SUBTEXT)
    ax1.set_xlabel(
        f'Showing last {last_n_bars} H1 bars  ·  '
        f'Key levels built from full {len(df):,}-bar dataset',
        color=SUBTEXT, fontsize=8, labelpad=8
    )

    # legend
    legend_items = [
        Line2D([0],[0], color=BLUE,   lw=1.5,         label='Close price (last 300 bars)'),
        Line2D([0],[0], color=RED,    lw=1.5, ls='--', label=f'POC  {levels.poc:.5f}'),
        Line2D([0],[0], color=GREEN,  lw=1.0, ls='--', label=f'VAH  {levels.vah:.5f}'),
        Line2D([0],[0], color=GREEN,  lw=1.0, ls='--', label=f'VAL  {levels.val:.5f}'),
        mpatches.Patch(color=GREEN, alpha=0.2,          label='Value Area (70% of volume)'),
        Line2D([0],[0], color=GREEN,  lw=1.0, ls=':',   label=f'HVN  ({len(levels.hvns)} levels)'),
        Line2D([0],[0], color=ORANGE, lw=1.0, ls=':',   label=f'LVN  ({len(levels.lvns)} targets)'),
    ]
    ax1.legend(handles=legend_items, loc='upper left',
               fontsize=7.5, facecolor='#1a1a2e',
               labelcolor=TEXT, framealpha=0.85,
               edgecolor='#2a2a3a')

    # ── VOLUME PROFILE HISTOGRAM — independent y axis ────────────
    profile    = levels.profile
    bar_height = (profile.index[1] - profile.index[0]) * 0.92

    bar_colors = []
    for price_level in profile.index:
        is_poc = abs(price_level - levels.poc) < 0.0008
        is_hvn = any(abs(price_level - h) < 0.0015 for h in levels.hvns)
        is_lvn = profile[price_level] <= profile.max() * Config.LVN_THRESHOLD
        in_va  = levels.val <= price_level <= levels.vah

        if is_poc:
            bar_colors.append(RED)
        elif is_hvn:
            bar_colors.append(GREEN)
        elif is_lvn:
            bar_colors.append(ORANGE)
        elif in_va:
            bar_colors.append('#1e3a2e')
        else:
            bar_colors.append(MUTED)

    ax2.barh(profile.index, profile.values,
             height=bar_height, color=bar_colors,
             alpha=0.9, zorder=3)

    # key level lines on profile
    ax2.axhline(levels.poc, color=RED,   linewidth=1.5, linestyle='--', alpha=0.9)
    ax2.axhline(levels.vah, color=GREEN, linewidth=1.0, linestyle='--', alpha=0.7)
    ax2.axhline(levels.val, color=GREEN, linewidth=1.0, linestyle='--', alpha=0.7)
    ax2.axhspan(levels.val, levels.vah, alpha=0.06, color=GREEN, zorder=0)

    # marker for current price on profile
    ax2.axhline(last_price, color=BLUE, linewidth=1.0,
                linestyle='-', alpha=0.8)
    ax2.annotate(f' Current\n {last_price:.5f}',
                 xy=(profile.max() * 0.6, last_price),
                 fontsize=6.5, color=BLUE, va='center')

    # profile covers full price range
    ax2.set_ylim(prof_ymin, prof_ymax)
    ax2.set_xlim(left=0)
    ax2.set_xlabel('Volume', color=SUBTEXT, fontsize=8, labelpad=8)
    ax2.set_ylabel('Full price range 2003→2026',
                   color=SUBTEXT, fontsize=7, labelpad=8)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.3f}')
    )
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', labelsize=7)
    ax2.tick_params(axis='x', labelsize=7, colors=SUBTEXT)
    ax2.set_title('23-Year\nVolume Profile',
                  color=SUBTEXT, fontsize=8, pad=8)

    # profile legend
    profile_legend = [
        mpatches.Patch(color=RED,      label=f'POC  {levels.poc:.5f}'),
        mpatches.Patch(color=GREEN,    label=f'HVN  (institutional)'),
        mpatches.Patch(color=ORANGE,   label=f'LVN  (fast move)'),
        mpatches.Patch(color='#1e3a2e',label=f'Value Area interior'),
        mpatches.Patch(color=MUTED,    label=f'Normal volume'),
    ]
    ax2.legend(handles=profile_legend,
               loc='upper right', fontsize=6.5,
               facecolor='#1a1a2e', labelcolor=TEXT,
               framealpha=0.85, edgecolor='#2a2a3a')

    # ── save ────────────────────────────────────────────────────
    import os
    os.makedirs("data/processed", exist_ok=True)
    outpath = f"data/processed/{symbol}_volume_profile.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
    log.info(f"Chart saved → {outpath}")
    plt.show()
    return fig
