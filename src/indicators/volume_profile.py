# src/indicators/volume_profile.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class VolumeProfileLevels:
    poc:  float          # Point of Control
    hvns: list[float]    # High Volume Nodes
    lvns: list[float]    # Low Volume Nodes
    profile: pd.Series   # full profile

def build_volume_profile(df: pd.DataFrame,
                         bins: int = Config.PROFILE_BINS) -> VolumeProfileLevels:
    """
    Build volume profile from OHLCV data.
    Distributes bar volume proportionally across its high-low range.
    """
    price_min = df['Low'].min()
    price_max = df['High'].max()

    price_levels   = np.linspace(price_min, price_max, bins)
    volume_at_price = np.zeros(bins)

    for _, row in df.iterrows():
        mask = (price_levels >= row['Low']) & (price_levels <= row['High'])
        if mask.sum() > 0:
            volume_at_price[mask] += row['Volume'] / mask.sum()

    profile = pd.Series(volume_at_price, index=price_levels)

    # --- Point of Control ---
    poc = float(profile.idxmax())

    # --- HVN / LVN ---
    vol_max = profile.max()
    hvns = list(profile[profile >= vol_max * Config.HVN_THRESHOLD].index)
    lvns = list(profile[profile <= vol_max * Config.LVN_THRESHOLD].index)

    log.info(f"Volume profile built | POC: {poc:.5f} | "
             f"HVNs: {len(hvns)} | LVNs: {len(lvns)}")

    return VolumeProfileLevels(poc=poc, hvns=hvns, lvns=lvns, profile=profile)


def price_near_level(price: float, level: float,
                     pip_size: float = 0.0001) -> bool:
    """Check if price is within POC_ZONE_PIPS of a level"""
    distance = abs(price - level) / pip_size
    return distance <= Config.POC_ZONE_PIPS