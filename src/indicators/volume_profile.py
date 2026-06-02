# src/indicators/volume_profile.py
"""
Volume Profile — Level Detection

Builds a price↔volume histogram from OHLCV data, then extracts:
  POC  — Point of Control (single highest-volume bin)
  VAH  — Value Area High  (top of 70% volume zone expanding from POC)
  VAL  — Value Area Low   (bottom of 70% volume zone)
  HVNs — High Volume Nodes as VolumeCluster objects with mini value areas
  LVNs — Low Volume Nodes  (local valleys — thin liquidity, price passes fast)

HVN/LVN Detection — Dual MA Crossover:
  Fixed-percentage methods miss valid clusters because they only capture
  the single highest peak.

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

Mini Value Areas:
  Each HVN cluster gets a 90% mini value area — the price range around the peak
  that contains 90% of the cluster's volume (HVN_VALUE_AREA_PCT). This gives the
  "width" of each cluster:
    - Wider zone = more volume = stronger level
    - Entry condition: price anywhere within [cluster.low, cluster.high], not just
      within ±5 pips of peak. Increases entry frequency without losing selectivity
      (rejection candle filter still applies).
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VolumeCluster:
    """
    A high-volume node with its mini value area.
    peak:   price of the highest-volume bin in this cluster zone
    low:    lower boundary of the 90% mini value area
    high:   upper boundary of the 90% mini value area
    volume: total volume in this cluster zone
    """
    peak:   float
    low:    float
    high:   float
    volume: float


@dataclass
class VolumeProfileLevels:
    poc:          float          # Point of Control — most traded price
    vah:          float          # Value Area High  — top of 70% volume zone
    val:          float          # Value Area Low   — bottom of 70% volume zone
    hvn_clusters: list           # list[VolumeCluster] — clusters with mini value areas
    hvns:         list           # list[float] — peak prices only (convenience)
    lvns:         list           # list[float] — Low Volume Node valley prices
    profile:      pd.Series      # full price→volume series


def price_near_level(price: float, level: float,
                     pip_size: float = 0.0001) -> bool:
    """Return True if price is within POC_ZONE_PIPS of a level."""
    distance = abs(price - level) / pip_size
    return distance <= Config.POC_ZONE_PIPS


def price_in_cluster(price: float, cluster: VolumeCluster) -> bool:
    """
    Return True if price is within the cluster's mini value area.
    Wider than price_near_level — uses actual cluster boundaries instead of
    a fixed ±5 pip zone. Entry can be anywhere the cluster has meaningful volume.
    """
    return cluster.low <= price <= cluster.high


def _compute_mini_value_area(zone_prices: np.ndarray,
                              zone_vols: np.ndarray,
                              target_pct: float) -> tuple[float, float]:
    """
    Compute a mini value area within a cluster zone.
    Expands from the peak outward until target_pct of the zone's volume is captured.
    Same algorithm as the standard POC→VAH/VAL expansion, applied per cluster.
    Returns (low_price, high_price).
    """
    n = len(zone_vols)
    if n == 1:
        return float(zone_prices[0]), float(zone_prices[0])

    peak_idx   = int(np.argmax(zone_vols))
    total_vol  = float(zone_vols.sum())
    target_vol = total_vol * target_pct

    accumulated = float(zone_vols[peak_idx])
    upper = peak_idx
    lower = peak_idx

    while accumulated < target_vol:
        can_up   = upper < n - 1
        can_down = lower > 0
        if not (can_up or can_down):
            break
        up_vol   = float(zone_vols[upper + 1]) if can_up   else 0.0
        down_vol = float(zone_vols[lower - 1]) if can_down else 0.0
        if up_vol >= down_vol and can_up:
            upper       += 1
            accumulated += up_vol
        elif can_down:
            lower       -= 1
            accumulated += down_vol
        else:
            upper       += 1
            accumulated += up_vol

    return float(zone_prices[lower]), float(zone_prices[upper])


def _merge_cluster_group(group: list) -> VolumeCluster:
    """Merge a group of adjacent VolumeCluster objects into one."""
    # Representative peak = peak of highest-volume member
    best  = max(group, key=lambda c: c.volume)
    low   = min(c.low  for c in group)
    high  = max(c.high for c in group)
    vol   = sum(c.volume for c in group)
    return VolumeCluster(peak=best.peak, low=low, high=high, volume=vol)


def _merge_clusters(clusters: list, merge_distance: float) -> list:
    """
    Merge VolumeCluster objects whose peaks are within merge_distance of each other.
    Merged cluster: outer boundaries of the group, peak from highest-volume member.
    """
    if not clusters:
        return []

    clusters = sorted(clusters, key=lambda c: c.peak)
    merged   = []
    group    = [clusters[0]]

    for c in clusters[1:]:
        if c.peak - group[-1].peak <= merge_distance:
            group.append(c)
        else:
            merged.append(_merge_cluster_group(group))
            group = [c]

    merged.append(_merge_cluster_group(group))
    return merged


def _merge_levels(prices: list, merge_distance: float) -> list:
    """
    Merge plain price levels within merge_distance of each other.
    Used for LVN valleys (no value area needed).
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
                     merge_distance: float,
                     value_area_pct: float) -> tuple[list, list]:
    """
    Dual trailing MA crossover to find HVN clusters and LVN valleys.

    HVNs returned as VolumeCluster objects with 90% mini value areas.
    LVNs returned as plain floats (valley prices).

    Falls back to simple percentile method if profile has fewer bins than
    2 × ma_period (too short for stable MAs).
    """
    n = len(profile)

    # ── Fallback for short profiles ───────────────────────────────
    if n < ma_period * 2:
        vols   = profile.values.astype(float)
        prices = profile.index.values
        vol_max = float(profile.max())
        hvn_px  = list(profile[profile >= vol_max * 0.60].index)
        lvn_px  = list(profile[profile <= vol_max * 0.20].index)
        # Mini value area: expand ±10 bins from each peak (local window only)
        clusters = []
        for px in hvn_px:
            peak_i = int(np.searchsorted(prices, px))
            peak_i = min(peak_i, n - 1)
            lo_i   = max(0, peak_i - 10)
            hi_i   = min(n, peak_i + 11)
            lo, hi = _compute_mini_value_area(
                prices[lo_i:hi_i], vols[lo_i:hi_i], value_area_pct
            )
            clusters.append(VolumeCluster(
                peak=float(prices[peak_i]), low=lo, high=hi,
                volume=float(vols[peak_i])
            ))
        return clusters, lvn_px

    vols   = profile.values.astype(float)
    prices = profile.index.values

    # ── Forward and backward trailing MAs ─────────────────────────
    fwd_series = pd.Series(vols).rolling(window=ma_period,
                                          min_periods=ma_period // 2).mean()
    bwd_series = (pd.Series(vols[::-1])
                  .rolling(window=ma_period, min_periods=ma_period // 2)
                  .mean()
                  .values[::-1])

    fwd = fwd_series.values
    bwd = bwd_series

    # ── Find crossovers ────────────────────────────────────────────
    diff  = fwd - bwd
    valid = ~(np.isnan(fwd) | np.isnan(bwd))

    diff_clean = np.where(valid, diff, 0.0)
    sign = np.sign(diff_clean)
    sign[sign == 0] = 1

    crossings = [0]
    for i in range(1, n):
        if sign[i] != sign[i - 1]:
            crossings.append(i)
    crossings.append(n)

    # ── Classify each zone and build clusters ──────────────────────
    hvn_clusters = []
    lvn_prices   = []

    for k in range(len(crossings) - 1):
        start = crossings[k]
        end   = crossings[k + 1]
        zone_vols   = vols[start:end]
        zone_prices = prices[start:end]

        if len(zone_vols) == 0:
            continue

        zone_sign = sign[start] if start < n else sign[start - 1]

        if zone_sign > 0:
            # HVN zone: compute peak + mini value area
            peak_idx    = int(np.argmax(zone_vols))
            peak_price  = float(zone_prices[peak_idx])
            lo, hi      = _compute_mini_value_area(zone_prices, zone_vols,
                                                    value_area_pct)
            zone_volume = float(zone_vols.sum())
            hvn_clusters.append(VolumeCluster(
                peak=peak_price, low=lo, high=hi, volume=zone_volume
            ))
        else:
            # LVN zone: just the minimum-volume price
            valley_idx = int(np.argmin(zone_vols))
            lvn_prices.append(float(zone_prices[valley_idx]))

    # ── Merge nearby clusters ──────────────────────────────────────
    hvn_clusters = _merge_clusters(hvn_clusters, merge_distance)
    lvn_prices   = _merge_levels(lvn_prices, merge_distance)

    return hvn_clusters, lvn_prices


def build_volume_profile(df: pd.DataFrame,
                         bins: int = Config.PROFILE_BINS,
                         value_area_pct: float = 0.70) -> VolumeProfileLevels:
    """
    Build a volume profile from OHLCV data.

    Distributes each bar's volume proportionally across its high-low range
    into `bins` price buckets. POC/VAH/VAL use the standard value area algorithm.
    HVN clusters use the dual MA crossover + 90% mini value area method.
    Works with any OHLCV timeframe — pass M15 for finer resolution.
    """
    price_min = float(df['Low'].min())
    price_max = float(df['High'].max())

    price_levels    = np.linspace(price_min, price_max, bins)
    volume_at_price = np.zeros(bins)

    lows  = df['Low'].values.astype(np.float64)
    highs = df['High'].values.astype(np.float64)
    vols  = df['Volume'].values.astype(np.float64)

    for i in range(len(df)):
        mask = (price_levels >= lows[i]) & (price_levels <= highs[i])
        n    = mask.sum()
        if n > 0:
            volume_at_price[mask] += vols[i] / n

    profile = pd.Series(volume_at_price, index=price_levels)

    # ── Point of Control ─────────────────────────────────────────
    poc = float(profile.idxmax())

    # ── Value Area ────────────────────────────────────────────────
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

    # ── HVN clusters + LVN valleys via dual MA crossover ─────────
    merge_dist = Config.CLUSTER_MERGE_PIPS * 0.0001
    hvn_clusters, lvns = _find_hvn_lvn_ma(
        profile,
        ma_period      = Config.HVN_MA_PERIOD,
        merge_distance = merge_dist,
        value_area_pct = Config.HVN_VALUE_AREA_PCT,
    )

    hvns = [c.peak for c in hvn_clusters]   # peak prices for backward compat

    log.debug(
        f"Volume profile  |  "
        f"POC: {poc:.5f}  VAH: {vah:.5f}  VAL: {val:.5f}  |  "
        f"HVN clusters: {len(hvn_clusters)}  LVNs: {len(lvns)}  |  "
        f"bars: {len(df):,}"
    )

    return VolumeProfileLevels(
        poc          = poc,
        vah          = vah,
        val          = val,
        hvn_clusters = hvn_clusters,
        hvns         = hvns,
        lvns         = lvns,
        profile      = profile
    )
