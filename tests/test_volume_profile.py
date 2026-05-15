# tests/test_volume_profile.py
import pandas as pd
import pytest
from src.indicators.volume_profile import build_volume_profile, price_near_level

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Open':   [1.1000, 1.1010, 1.1005, 1.0990, 1.1015],
        'High':   [1.1020, 1.1025, 1.1015, 1.1010, 1.1030],
        'Low':    [1.0990, 1.1000, 1.0995, 1.0980, 1.1000],
        'Close':  [1.1010, 1.1005, 1.1000, 1.1005, 1.1020],
        'Volume': [100, 150, 120, 200, 180]
    })

def test_poc_within_range(sample_df):
    levels = build_volume_profile(sample_df)
    assert sample_df['Low'].min() <= levels.poc <= sample_df['High'].max()

def test_hvns_not_empty(sample_df):
    levels = build_volume_profile(sample_df)
    assert len(levels.hvns) > 0

def test_price_near_level():
    assert price_near_level(1.10005, 1.10000) == True
    assert price_near_level(1.1050, 1.1000) == False