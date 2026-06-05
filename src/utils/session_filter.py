# src/utils/session_filter.py
from datetime import datetime, time
import pytz
from src.config import Config

UTC = pytz.UTC
NY  = pytz.timezone('America/New_York')

SESSIONS = {
    'sydney':  (time(17, 0), time(2,  0)),   # 5pm - 2am ET
    'tokyo':   (time(19, 0), time(4,  0)),   # 7pm - 4am ET
    'london':  (time(3,  0), time(12, 0)),   # 3am - 12pm ET  ← best
    'new_york':(time(8,  0), time(17, 0)),   # 8am - 5pm ET   ← best
}

def get_active_session(dt: datetime) -> str | None:
    """Return which session is active — London and NY overlap is peak"""
    ny_time = dt.astimezone(NY).time()

    # London/NY overlap 8am-12pm ET — the golden window
    if time(8, 0) <= ny_time <= time(12, 0):
        return 'london_ny_overlap'
    elif time(3, 0) <= ny_time <= time(12, 0):
        return 'london'
    elif time(8, 0) <= ny_time <= time(17, 0):
        return 'new_york'
    else:
        return None   # don't trade

def is_asian_session(ny_time_t: time) -> bool:
    """
    Asian (Sydney/Tokyo) session in ET — roughly 5pm → 3am, i.e. the hours OUTSIDE
    the London/NY window (3am-5pm). This is where AUD/NZD/JPY do their real volume.
    """
    return ny_time_t >= time(17, 0) or ny_time_t < time(3, 0)


def is_tradeable_session(dt: datetime) -> bool:
    """
    Returns True during tradeable sessions, Mon-Fri.
    Default: London or NY only. If Config.INCLUDE_ASIAN_SESSION is set, also allow the
    Asian session — relevant for AUD/NZD/JPY whose liquidity peaks in Tokyo hours.
    """
    try:
        ny_time = dt.astimezone(NY)
    except Exception:
        return True

    # block weekends entirely
    if ny_time.weekday() >= 5:
        return False

    if get_active_session(dt) is not None:
        return True

    if getattr(Config, "INCLUDE_ASIAN_SESSION", False) and is_asian_session(ny_time.time()):
        return True

    return False
