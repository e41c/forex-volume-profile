# src/utils/news_filter.py
"""
High impact news events destroy technical setups.
This filter blocks trading 30 minutes before and after
any high-impact event.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Free forex factory calendar API
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

def get_high_impact_events() -> list[datetime]:
    """Fetch this week's high impact USD/EUR news events"""
    try:
        response = requests.get(CALENDAR_URL, timeout=5)
        events   = response.json()
        high_impact = [
            pd.to_datetime(e['date']).tz_localize('UTC')
            for e in events
            if e.get('impact') == 'High'
            and e.get('country') in ['USD', 'EUR']
        ]
        return high_impact
    except Exception:
        return []   # if API fails, assume no news (fail open)

def is_news_window(dt: datetime,
                   buffer_minutes: int = 30) -> bool:
    """Return True if we are within buffer_minutes of a high impact event"""
    events = get_high_impact_events()
    buffer = timedelta(minutes=buffer_minutes)
    for event_time in events:
        if abs(dt - event_time) <= buffer:
            return True
    return False