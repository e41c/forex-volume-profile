# src/indicators/volume_profile.py
"""
Volume Profile — Level Detection

Builds a price↔volume histogram from OHLCV data, then extracts:
  POC  — Point of Control (single highest-volume bin)
  VAH  — Value Area High  (top of 70% volume zone expanding from POC)
  VAL  — Value Area Low   (bottom of 70% volume zone)
  HVNs — High Volume Nodes (local peaks in the histogram)
  LVNs — Low Volume Nodes  (local valleys — thin liquidity, price passes through fast)

HVN/LVN Detection — Dual MA Crossover:
  Fixed-percentage methods (e.g. "volume ≥ 70% of max") miss many valid clusters
  because they only capture the single highest peak.

  Instead: compute two trailing MAs of the volume histogram, one scanning
  bottom-to-top and one scanning top-to-bottom. Where they cross marks a
  structural peak or valley — every meaningful cluster, not just the dominant one.

  Forward MA (fwd[i]): trailing mean of bins [i-period : i]
                       "knows what's below price level i"
  Backward MA (bwd[i]): trailing mean of bins [i : i+period]
                        "knows what's above price level i"

  Transition fwd < bwd → fwd > bwd (going up through price): just passed an HVN peak
  Transition fwd > bwd → fwd < bwd (going up through price): just passed an LVN valley

  The actual peak/valley price = max/min volume between the two surrounding crossings.
  Clusters within CLUSTER_MERGE_PIPS of each other are merged (deduplication).

  Algorithm credit: original MQL4 implementation by user, ported to Python.
  Parameters: MA period = 55 (user-validated), merge threshold = CLUSTER_MERGE_PIPS.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VolumeProfileLevels:
    poc:     float       # Point of Control — most traded price
    vah:     float       # Value Area High  — top of 70% volume zone
    val:     float       # Value Area Low   — bottom of 70% volume zone
    hvns:    list        # High Volume Nodes (peaks in the histogram)
    lvns:    list        # Low Volume Nodes  (valleys — thin liquidity)
    profile: pd.Series   # full price→volume series


def _merge_levels(prices: list, merge_distance: float) -> list:
    """
    Merge price levels within merge_distance of each other.
    Consecutive levels that are close are averaged into one.
    """
    if not prices:
        return []

    sorted_prices = sorted(prices)
    merged = []
    group  = [sorted_prices[0]]

    for p in sorted_prices[1:]:
        if p - group[-1] <= merge_distance:
            group.append(p)
        else:
            merged.append(float(np.mean(group)))
            group = [p]

    merged.append(float(np.mean(group)))
    return merged


def _find_hvn_lvn_ma(profile: pd.Series,
                     ma_period: int,
                     merge_distance: float) -> tuple[list, list]:
    """
    Dual trailing MA crossover to find HVN peaks and LVN valleys.

    See module docstring for algorithm explanation.
    Falls back to simple percentile method if profile has fewer bins than
    2 × ma_period (too short for stable MAs).
    """
    n = len(profile)

    # ── Fallback for short profiles ───────────────────────────────
    if n < ma_period * 2:
        vol_max = float(profile.max())
        hvns = list(profile[profile >= vol_max * 0.60].index)
        lvns = list(profile[profile <= vol_max * 0.20].index)
        return hvns, lvns

    vols   = profile.values.astype(float)
    prices = profile.index.values

    # ── Forward and backward trailing MAs ─────────────────────────
    # fwd[i] = mean(vols[i-period : i])  — trailing from below
    # bwd[i] = mean(vols[i : i+period])  — trailing from above
    #        = reversed trailing MA of the reversed array, then reversed back

    fwd_series = pd.Series(vols).rolling(window=ma_period, min_periods=ma_period // 2).mean()
    bwd_series = (pd.Series(vols[::-1])
                  .rolling(window=ma_period, min_periods=ma_period // 2)
                  .mean()
                  .values[::-1])

    fwd = fwd_series.values
    bwd = bwd_series

    # ── Find crossovers ────────────────────────────────────────────
    diff  = fwd - bwd
    valid = ~(np.isnan(fwd) | np.isnan(bwd))

    # Treat NaN edges as zero diff — they bracket the valid region
    diff_clean = np.where(valid, diff, 0.0)
    sign = np.sign(diff_clean)
    sign[sign == 0] = 1  # treat exact-zero as positive to avoid spurious crossings

    crossings = [0]  # start at first bin
    for i in range(1, n):
        if sign[i] != sign[i - 1]:
            crossings.append(i)
    crossings.append(n)  # end at last bin

    # ── Classify each zone between crossings ───────────────────────
    hvn_prices = []
    lvn_prices = []

    for k in range(len(crossings) - 1):
        start = crossings[k]
        end   = crossings[k + 1]
        zone_vols   = vols[start:end]
        zone_prices = prices[start:end]

        if len(zone_vols) == 0:
            continue

        # Sign of diff at the START of this zone:
        #   diff > 0 (fwd > bwd): more vol below than above → just exited a peak → HVN zone
        #   diff < 0 (fwd < bwd): more vol above than below → approaching a peak → LVN zone
        zone_sign = sign[start] if start < n else sign[start - 1]

        if zone_sign > 0:
            # HVN: find the maximum volume bin in this zone
            peak_idx = np.argmax(zone_vols)
            hvn_prices.append(float(zone_prices[peak_idx]))
        else:
            # LVN: find the minimum volume bin in this zone
            valley_idx = np.argmin(zone_vols)
            lvn_prices.append(float(zone_prices[valley_idx]))

    # ── Merge nearby clusters ──────────────────────────────────────
    hvn_prices = _merge_levels(hvn_prices, merge_distance)
    lvn_prices = _merge_levels(lvn_prices, merge_distance)

    return hvn_prices, lvn_prices


def build_volume_profile(df: pd.DataFrame,
                         bins: int = Config.PROFILE_BINS,
                         value_area_pct: float = 0.70) -> VolumeProfileLevels:
    """
    Build a volume profile from OHLCV data.

    Distributes each bar's volume proportionally across its high-low range
    into `bins` price buckets. POC/VAH/VAL use the standard value area algorithm.
    HVN/LVN use the dual MA crossover method — more sensitive than fixed thresholds.
    """
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())

    price_levels    = np.linspace(price_min, price_max, bins)
    volume_at_price = np.zeros(bins)

    # Pull arrays out of DataFrame once — avoids repeated pandas overhead
    lows  = df['Low'].values.astype(np.float64)
    highs = df['High'].values.astype(np.float64)
    vols  = df['Volume'].values.astype(np.float64)

    # Vectorized inner loop — distributes each bar's volume across its price range
    for i in range(len(df)):
        mask = (price_levels >= lows[i]) & (price_levels <= highs[i])
        n    = mask.sum()
        if n > 0:
            volume_at_price[mask] += vols[i] / n

    profile = pd.Series(volume_at_price, index=price_levels)

    # ── Point of Control ─────────────────────────────────────────
    poc = float(profile.idxmax())

    # ── Value Area (expanding from POC until 70% of volume is captured) ──
    total_volume  = profile.sum()
    target_volume = total_volume * value_area_pct
    indices       = list(profile.index)
    poc_pos       = indices.index(poc)

    accumulated = float(profile[poc])
    upper       = poc_pos
    lower       = poc_pos

    while accumulated < target_volume:
        can_up   = upper < len(indices) - 1
        can_down = lower > 0
        up_vol   = float(profile[indices[upper + 1]]) if can_up   else 0.0
        down_vol = float(profile[indices[lower - 1]]) if can_down else 0.0

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

    # ── HVN / LVN via dual MA crossover ──────────────────────────
    merge_dist = Config.CLUSTER_MERGE_PIPS * 0.0001  # pips → price units
    hvns, lvns = _find_hvn_lvn_ma(
        profile,
        ma_period      = Config.HVN_MA_PERIOD,
        merge_distance = merge_dist,
    )

    log.debug(
        f"Volume profile  |  "
        f"POC: {poc:.5f}  VAH: {vah:.5f}  VAL: {val:.5f}  |  "
        f"HVNs: {len(hvns)}  LVNs: {len(lvns)}  |  "
        f"bars: {len(df):,}"
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
    """Return True if price is within POC_ZONE_PIPS of a level."""
    distance = abs(price - level) / pip_size
    return distance <= Config.POC_ZONE_PIPS
