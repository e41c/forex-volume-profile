# src/visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from src.indicators.volume_profile import build_volume_profile, VolumeProfileLevels
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

BG       = '#0f0f1a'
PANEL    = '#16162a'
BLUE     = '#378ADD'
RED      = '#E24B4A'
GREEN    = '#1D9E75'
ORANGE   = '#D85A30'
MUTED    = '#2a2a4a'
TEXT     = '#c8c8c4'
SUBTEXT  = '#888780'


def calculate_value_area(profile: pd.Series,
                          target_pct: float = 0.70):
    """
    Calculate Value Area High (VAH) and Value Area Low (VAL).
    The value area contains target_pct (default 70%) of total volume.
    """
    total_volume  = profile.sum()
    target_volume = total_volume * target_pct
    poc_idx       = profile.idxmax()

    accumulated = profile[poc_idx]
    indices     = list(profile.index)
    poc_pos     = indices.index(poc_idx)

    upper = poc_pos
    lower = poc_pos

    while accumulated < target_volume:
        can_go_up   = upper < len(indices) - 1
        can_go_down = lower > 0

        up_vol   = profile[indices[upper + 1]] if can_go_up   else 0
        down_vol = profile[indices[lower - 1]] if can_go_down else 0

        if not can_go_up and not can_go_down:
            break

        if up_vol >= down_vol and can_go_up:
            upper       += 1
            accumulated += up_vol
        elif can_go_down:
            lower       -= 1
            accumulated += down_vol
        else:
            upper       += 1
            accumulated += up_vol

    vah = indices[upper]
    val = indices[lower]
    return float(vah), float(val)


def plot_volume_profile(df: pd.DataFrame,
                        levels: VolumeProfileLevels,
                        symbol: str = Config.SYMBOL,
                        last_n_bars: int = 300):

    vah, val = calculate_value_area(levels.profile)

    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    fig.suptitle(
        f'{symbol}  —  Volume Profile Analysis',
        color=TEXT, fontsize=14, fontweight='normal',
        y=0.97, x=0.42
    )

    gs  = gridspec.GridSpec(
        1, 2,
        width_ratios=[3, 1],
        wspace=0.0,
        left=0.06, right=0.97,
        top=0.92,  bottom=0.09
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    for ax in [ax1, ax2]:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3a')

    plot_df = df.tail(last_n_bars).copy()
    ymin    = plot_df['Low'].min()  * 0.9985
    ymax    = plot_df['High'].max() * 1.0015

    # ── value area shading ──────────────────────────────────────────
    ax1.axhspan(val, vah, alpha=0.08, color=GREEN, zorder=0,
                label=f'Value Area  {val:.5f} – {vah:.5f}')

    # ── price line ──────────────────────────────────────────────────
    ax1.plot(plot_df.index, plot_df['Close'],
             color=BLUE, linewidth=1.0, zorder=3, label='Close price')

    # ── POC ─────────────────────────────────────────────────────────
    ax1.axhline(levels.poc, color=RED, linewidth=1.5,
                linestyle='--', alpha=0.95, zorder=4,
                label=f'POC  {levels.poc:.5f}')

    # ── VAH / VAL ───────────────────────────────────────────────────
    ax1.axhline(vah, color=GREEN, linewidth=1.0,
                linestyle='--', alpha=0.8, zorder=4,
                label=f'VAH  {vah:.5f}')
    ax1.axhline(val, color=GREEN, linewidth=1.0,
                linestyle='--', alpha=0.8, zorder=4,
                label=f'VAL  {val:.5f}')

    # ── HVN lines (only those visible in current price window) ──────
    visible_hvns = [h for h in levels.hvns if ymin < h < ymax]
    for i, hvn in enumerate(visible_hvns):
        ax1.axhline(hvn, color=GREEN, linewidth=0.5,
                    linestyle=':', alpha=0.5, zorder=2,
                    label='HVN' if i == 0 else '_')

    # ── LVN lines (only top 5 visible) ──────────────────────────────
    visible_lvns = sorted(
        [l for l in levels.lvns if ymin < l < ymax],
        key=lambda x: abs(x - plot_df['Close'].iloc[-1])
    )[:5]
    for i, lvn in enumerate(visible_lvns):
        ax1.axhline(lvn, color=ORANGE, linewidth=0.5,
                    linestyle=':', alpha=0.4, zorder=2,
                    label='LVN (target zone)' if i == 0 else '_')

    # ── current price label ─────────────────────────────────────────
    last_price = plot_df['Close'].iloc[-1]
    ax1.annotate(
        f'  {last_price:.5f}',
        xy=(plot_df.index[-1], last_price),
        fontsize=8, color=BLUE, va='center',
        xytext=(5, 0), textcoords='offset points'
    )

    # ── axis labels ─────────────────────────────────────────────────
    ax1.set_ylabel('Price', color=SUBTEXT, fontsize=9, labelpad=8)
    ax1.set_ylim(ymin, ymax)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:.4f}')
    )
    ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax1.xaxis.get_majorticklabels(),
             rotation=30, ha='right', color=SUBTEXT)

    # ── subtitle with date range ─────────────────────────────────────
    date_start = df.index[0].strftime('%b %Y')
    date_end   = df.index[-1].strftime('%b %Y')
    ax1.set_xlabel(
        f'Showing last {last_n_bars} bars  ·  '
        f'Full dataset: {date_start} → {date_end}  ·  '
        f'{len(df):,} total bars',
        color=SUBTEXT, fontsize=8, labelpad=8
    )

    # ── legend ──────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0],[0], color=BLUE,   lw=1.5,              label='Close price'),
        Line2D([0],[0], color=RED,    lw=1.5, ls='--',     label=f'POC  {levels.poc:.5f}'),
        Line2D([0],[0], color=GREEN,  lw=1.0, ls='--',     label=f'VAH  {vah:.5f}'),
        Line2D([0],[0], color=GREEN,  lw=1.0, ls='--',     label=f'VAL  {val:.5f}'),
        mpatches.Patch(color=GREEN,   alpha=0.15,           label='Value Area (70% vol)'),
        Line2D([0],[0], color=GREEN,  lw=1.0, ls=':',      label=f'HVN  ({len(levels.hvns)} levels)'),
        Line2D([0],[0], color=ORANGE, lw=1.0, ls=':',      label=f'LVN  ({len(levels.lvns)} targets)'),
    ]
    ax1.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=7.5,
        facecolor='#1a1a2e',
        labelcolor=TEXT,
        framealpha=0.85,
        edgecolor='#2a2a3a',
        ncol=1
    )

    # ════════════════════════════════════════════════════════════════
    # VOLUME PROFILE HISTOGRAM
    # ════════════════════════════════════════════════════════════════
    profile    = levels.profile
    bar_height = (profile.index[1] - profile.index[0]) * 0.92
    bar_colors = []

    for price_level in profile.index:
        is_poc = abs(price_level - levels.poc)   < 0.0008
        is_hvn = any(abs(price_level - h) < 0.0015 for h in levels.hvns)
        is_lvn = profile[price_level] <= profile.max() * Config.LVN_THRESHOLD
        in_va  = val <= price_level <= vah

        if is_poc:
            bar_colors.append(RED)
        elif is_hvn:
            bar_colors.append(GREEN)
        elif is_lvn:
            bar_colors.append(ORANGE)
        elif in_va:
            bar_colors.append('#234434')
        else:
            bar_colors.append(MUTED)

    ax2.barh(
        profile.index,
        profile.values,
        height=bar_height,
        color=bar_colors,
        alpha=0.9,
        zorder=3
    )

    # value area shading on profile
    ax2.axhspan(val, vah, alpha=0.06, color=GREEN, zorder=0)

    # POC line on profile
    ax2.axhline(levels.poc, color=RED,   linewidth=1.5, linestyle='--', alpha=0.9)
    ax2.axhline(vah,        color=GREEN, linewidth=1.0, linestyle='--', alpha=0.7)
    ax2.axhline(val,        color=GREEN, linewidth=1.0, linestyle='--', alpha=0.7)

    ax2.set_xlabel('Volume', color=SUBTEXT, fontsize=8, labelpad=8)
    ax2.tick_params(labelleft=False)
    ax2.tick_params(axis='x', labelsize=7, colors=SUBTEXT)
    ax2.set_xlim(left=0)

    # profile legend
    profile_legend = [
        mpatches.Patch(color=RED,    label='POC — most traded price'),
        mpatches.Patch(color=GREEN,  label='HVN — institutional zone'),
        mpatches.Patch(color=ORANGE, label='LVN — fast-move target'),
        mpatches.Patch(color='#234434', label='Value area interior'),
        mpatches.Patch(color=MUTED,  label='Normal volume'),
    ]
    ax2.legend(
        handles=profile_legend,
        loc='upper right',
        fontsize=7,
        facecolor='#1a1a2e',
        labelcolor=TEXT,
        framealpha=0.85,
        edgecolor='#2a2a3a'
    )

    ax2.set_title('Volume\nProfile',
                  color=SUBTEXT, fontsize=8, pad=8)

    # ── save ────────────────────────────────────────────────────────
    import os
    os.makedirs("data/processed", exist_ok=True)
    outpath = f"data/processed/{symbol}_volume_profile.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor=BG)
    log.info(f"Chart saved → {outpath}")

    plt.show()
    return fig
