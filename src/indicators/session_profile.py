# src/indicators/session_profile.py
"""
Priority 4 — Rolling Session Profile

Instead of one giant 23-year profile, this builds profiles for:
- Daily session   (last 24 hours)
- Weekly session  (last 5 days)
- Monthly session (last 22 trading days)

Two POCs working together = much stronger signal:
  23yr POC  = 1.13093  ← major institutional memory
  Today POC = 1.16100  ← where institutions are trading RIGHT NOW
  If price rejects from today's POC AND it's near 23yr POC = very strong signal
"""
import pandas as pd
from dataclasses import dataclass
from src.indicators.volume_profile import (
    build_volume_profile,
    VolumeProfileLevels
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# how many H1 bars make each session window
SESSION_BARS = {
    "daily":   24,    # last 24 H1 bars = 1 trading day
    "weekly":  120,   # last 120 H1 bars = 5 trading days
    "monthly": 480,   # last 480 H1 bars = ~22 trading days
}


@dataclass
class MultiSessionLevels:
    long_term: VolumeProfileLevels   # last 2000 H1 bars (~83 days / 3 months)
    daily:     VolumeProfileLevels   # last 24 bars (~1 trading day)
    weekly:    VolumeProfileLevels   # last 120 bars (~5 trading days)
    monthly:   VolumeProfileLevels   # last 480 bars (~22 trading days)


def build_session_profile(df: pd.DataFrame,
                          session: str = "daily",
                          bins: int = 50,
                          pip_size: float = 0.0001) -> VolumeProfileLevels:
    """
    Build a volume profile for a specific rolling session window.

    Args:
        df:       full OHLCV dataframe
        session:  one of 'daily', 'weekly', 'monthly'
        bins:     price bucket resolution (lower than long-term is fine)
        pip_size: instrument pip size, threaded through for correct cluster merging
    """
    bars = SESSION_BARS.get(session)
    if bars is None:
        raise ValueError(f"Unknown session: {session}. "
                         f"Use: {list(SESSION_BARS.keys())}")

    if len(df) < bars:
        log.warning(
            f"Not enough bars for {session} profile "
            f"(need {bars}, have {len(df)}) — using all available"
        )
        session_df = df
    else:
        session_df = df.tail(bars)

    log.debug(
        f"Building {session} profile on {len(session_df)} bars  "
        f"({session_df.index[0].strftime('%Y-%m-%d %H:%M')} → "
        f"{session_df.index[-1].strftime('%Y-%m-%d %H:%M')})"
    )

    return build_volume_profile(session_df, bins=bins, pip_size=pip_size)


def build_multi_session_levels(df: pd.DataFrame,
                               pip_size: float = 0.0001) -> MultiSessionLevels:
    """
    Build all four profiles at once using fixed-bin resolution (proven best).
    Long-term uses the full passed window; sessions use rolling sub-windows.
    pip_size is threaded through so cluster merging is correct across pairs.
    """
    log.debug("Building multi-session volume profiles...")

    long_term = build_volume_profile(df, bins=100,            pip_size=pip_size)
    daily     = build_session_profile(df, "daily",   bins=50,  pip_size=pip_size)
    weekly    = build_session_profile(df, "weekly",  bins=75,  pip_size=pip_size)
    monthly   = build_session_profile(df, "monthly", bins=100, pip_size=pip_size)

    log.debug(
        f"Multi-session profiles built:\n"
        f"  Long-term  POC: {long_term.poc:.5f}  "
        f"VAH: {long_term.vah:.5f}  VAL: {long_term.val:.5f}\n"
        f"  Monthly    POC: {monthly.poc:.5f}  "
        f"VAH: {monthly.vah:.5f}  VAL: {monthly.val:.5f}\n"
        f"  Weekly     POC: {weekly.poc:.5f}  "
        f"VAH: {weekly.vah:.5f}  VAL: {weekly.val:.5f}\n"
        f"  Daily      POC: {daily.poc:.5f}  "
        f"VAH: {daily.vah:.5f}  VAL: {daily.val:.5f}"
    )

    return MultiSessionLevels(
        long_term = long_term,
        daily     = daily,
        weekly    = weekly,
        monthly   = monthly,
    )


def poc_confluence(levels: MultiSessionLevels,
                   pip_size: float = 0.0001,
                   pip_threshold: float = 20.0) -> dict:
    """
    Check if multiple session POCs are within pip_threshold of each other.
    When 2+ POCs cluster together = extremely strong level.

    Returns a dict of which POCs are confluent and the average price.
    """
    pocs = {
        "long_term": levels.long_term.poc,
        "monthly":   levels.monthly.poc,
        "weekly":    levels.weekly.poc,
        "daily":     levels.daily.poc,
    }

    confluent = {}
    checked   = set()

    for name_a, poc_a in pocs.items():
        for name_b, poc_b in pocs.items():
            if name_a == name_b or (name_b, name_a) in checked:
                continue
            checked.add((name_a, name_b))
            distance = abs(poc_a - poc_b) / pip_size
            if distance <= pip_threshold:
                confluent[f"{name_a}+{name_b}"] = {
                    "price_a":    poc_a,
                    "price_b":    poc_b,
                    "avg_price":  round((poc_a + poc_b) / 2, 5),
                    "distance_pips": round(distance, 1),
                }

    if confluent:
        log.debug(f"POC confluence detected: {list(confluent.keys())}")
    else:
        log.debug("No POC confluence found at current levels")

    return confluent
