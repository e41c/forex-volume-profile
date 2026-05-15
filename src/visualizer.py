# src/visualizer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.indicators.volume_profile import build_volume_profile, VolumeProfileLevels
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

def plot_volume_profile(df: pd.DataFrame, levels: VolumeProfileLevels,
                        symbol: str = Config.SYMBOL,
                        last_n_bars: int = 200):

    fig = plt.figure(figsize=(16, 8), facecolor='#1a1a2e')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.02)

    ax1 = fig.add_subplot(gs[0])  # price chart
    ax2 = fig.add_subplot(gs[1])  # volume profile

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#888', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')

    # --- Price chart ---
    plot_df = df.tail(last_n_bars)
    ax1.plot(plot_df.index, plot_df['Close'],
             color='#4a9eff', linewidth=1.0, label='Close')

    # POC line
    ax1.axhline(y=levels.poc, color='#ff4444', linewidth=1.5,
                linestyle='--', alpha=0.9, label=f'POC {levels.poc:.5f}')

    # HVN zones
    for i, hvn in enumerate(levels.hvns):
        ax1.axhline(y=hvn, color='#00c896', linewidth=0.6,
                    linestyle=':', alpha=0.6,
                    label='HVN' if i == 0 else '')

    # LVN zones (just a few so chart stays clean)
    visible_lvns = [l for l in levels.lvns
                    if plot_df['Low'].min() < l < plot_df['High'].max()][:5]
    for i, lvn in enumerate(visible_lvns):
        ax1.axhline(y=lvn, color='#ff8c42', linewidth=0.5,
                    linestyle=':', alpha=0.4,
                    label='LVN' if i == 0 else '')

    ax1.set_title(f'{symbol} — Volume Profile Analysis',
                  color='white', fontsize=13, pad=12)
    ax1.set_ylabel('Price', color='#888', fontsize=9)
    ax1.legend(loc='upper left', fontsize=8,
               facecolor='#252540', labelcolor='white', framealpha=0.7)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(8))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # --- Volume profile histogram ---
    profile = levels.profile
    bar_colors = []
    for price_level, vol in profile.items():
        if abs(price_level - levels.poc) < 0.0008:
            bar_colors.append('#ff4444')     # POC — red
        elif any(abs(price_level - h) < 0.0012 for h in levels.hvns):
            bar_colors.append('#00c896')     # HVN — green
        elif vol <= profile.max() * Config.LVN_THRESHOLD:
            bar_colors.append('#ff8c42')     # LVN — orange
        else:
            bar_colors.append('#2a2a4a')     # normal — dark

    ax2.barh(profile.index, profile.values,
             height=(profile.index[1] - profile.index[0]) * 0.9,
             color=bar_colors, alpha=0.85)

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel('Volume', color='#888', fontsize=9)
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', labelsize=7)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.set_title('Volume\nProfile', color='white', fontsize=9, pad=12)

    # --- Legend for profile ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff4444', label=f'POC {levels.poc:.5f}'),
        Patch(facecolor='#00c896', label=f'HVN ({len(levels.hvns)})'),
        Patch(facecolor='#ff8c42', label=f'LVN ({len(levels.lvns)})'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right',
               fontsize=7, facecolor='#252540',
               labelcolor='white', framealpha=0.7)

    plt.tight_layout()

    outpath = f"data/processed/{symbol}_volume_profile.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e')
    log.info(f"Chart saved to {outpath}")

    plt.show()
    return fig