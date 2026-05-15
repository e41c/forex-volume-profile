# src/data/dukascopy_provider.py
from datetime import datetime
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD
from .base_provider import BaseDataProvider

INSTRUMENT_MAP = {
    "EURUSD": INSTRUMENT_FX_MAJORS_EUR_USD,
    # add more pairs as you need them
}

INTERVAL_MAP = {
    "M1":  dukascopy_python.INTERVAL_MINUTE_1,
    "M5":  dukascopy_python.INTERVAL_MINUTE_5,
    "H1":  dukascopy_python.INTERVAL_HOUR_1,
    "D1":  dukascopy_python.INTERVAL_DAY_1,
}

class DukascopyProvider(BaseDataProvider):

    def get_ohlcv(self, symbol: str, timeframe: str, 
                  start: datetime, end: datetime):
        
        df = dukascopy_python.fetch(
            instrument   = INSTRUMENT_MAP[symbol],
            interval     = INTERVAL_MAP[timeframe],
            offer_side   = dukascopy_python.OFFER_SIDE_BID,
            start        = start,
            end          = end,
        )
        return df