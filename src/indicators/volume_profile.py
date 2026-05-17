# src/indicators/volume_profile.py
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VolumeProfileLevels:
    poc:     float          # Point of Control — most traded price
    vah:     float          # Value Area High  — top of 70% volume zone
    val:     float          # Value Area Low   — bottom of 70% volume zone
    hvns:    list           # High Volume Nodes
    lvns:    list           # Low Volume Nodes
    profile: pd.Series      # full price→volume series


def build_volume_profile(df: pd.DataFrame,
                         bins: int = Config.PROFILE_BINS,
                         value_area_pct: float = 0.70) -> VolumeProfileLevels:
    """
    Build a volume profile from OHLCV data.
    Distributes each bar's volume proportionally across its high-low range.
    """
    price_min = df['Low'].min()
    price_max = df['High'].max()

    price_levels    = np.linspace(price_min, price_max, bins)
    volume_at_price = np.zeros(bins)

    for _, row in df.iterrows():
        mask = (price_levels >= row['Low']) & (price_levels <= row['High'])
        if mask.sum() > 0:
            volume_at_price[mask] += row['Volume'] / mask.sum()

    profile = pd.Series(volume_at_price, index=price_levels)

    # ── Point of Control ──────────────────────────────────────────
    poc = float(profile.idxmax())

    # ── Value Area (70% of volume around POC) ────────────────────
    total_volume  = profile.sum()
    target_volume = total_volume * value_area_pct
    indices       = list(profile.index)
    poc_pos       = indices.index(poc)

    accumulated = profile[poc]
    upper       = poc_pos
    lower       = poc_pos

    while accumulated < target_volume:
        can_up   = upper < len(indices) - 1
        can_down = lower > 0
        up_vol   = profile[indices[upper + 1]] if can_up   else 0
        down_vol = profile[indices[lower - 1]] if can_down else 0

        if not can_up and not can_down:
            break
        if up_vol >= down_vol and can_up:
            upper       += 1
            accumulated += up_vol
        elif can_down:
            lower       -= 1
            accumulated += down_vol
        else:
            upper       += 1
            accumulated += up_vol

    vah = float(indices[upper])
    val = float(indices[lower])

    # ── HVN / LVN ────────────────────────────────────────────────
    vol_max = profile.max()
    hvns    = list(profile[profile >= vol_max * Config.HVN_THRESHOLD].index)
    lvns    = list(profile[profile <= vol_max * Config.LVN_THRESHOLD].index)

    log.info(
        f"Volume profile built  |  "
        f"POC: {poc:.5f}  VAH: {vah:.5f}  VAL: {val:.5f}  |  "
        f"HVNs: {len(hvns)}  LVNs: {len(lvns)}"
    )

    return VolumeProfileLevels(
        poc     = poc,
        vah     = vah,
        val     = val,
        hvns    = hvns,
        lvns    = lvns,
        profile = profile
    )


def price_near_level(price: float, level: float,
                     pip_size: float = 0.0001) -> bool:
    """Return True if price is within POC_ZONE_PIPS of a level"""
    distance = abs(price - level) / pip_size
    return distance <= Config.POC_ZONE_PIPS
