# src/data/dukascopy_provider.py
import pandas as pd
from datetime import datetime
import dukascopy_python
from dukascopy_python import instruments
from .base_provider import BaseDataProvider
from src.utils.logger import get_logger

log = get_logger(__name__)

INSTRUMENT_MAP = {
    "EURUSD": instruments.INSTRUMENT_FX_MAJORS_EUR_USD,
    "GBPUSD": instruments.INSTRUMENT_FX_MAJORS_GBP_USD,
    "USDJPY": instruments.INSTRUMENT_FX_MAJORS_USD_JPY,
    "AUDUSD": instruments.INSTRUMENT_FX_MAJORS_AUD_USD,
    "USDCAD": instruments.INSTRUMENT_FX_MAJORS_USD_CAD,
    "USDCHF": instruments.INSTRUMENT_FX_MAJORS_USD_CHF,
}

INTERVAL_MAP = {
    "M1":  dukascopy_python.INTERVAL_MIN_1,
    "M5":  dukascopy_python.INTERVAL_MIN_5,
    "M15": dukascopy_python.INTERVAL_MIN_15,
    "H1":  dukascopy_python.INTERVAL_HOUR_1,
    "D1":  dukascopy_python.INTERVAL_DAY_1,
}

class DukascopyProvider(BaseDataProvider):

    def get_ohlcv(self, symbol: str, timeframe: str,
                  start: datetime = None, end: datetime = None,
                  **kwargs) -> pd.DataFrame:

        log.info(f"Fetching {symbol} {timeframe} from Dukascopy")

        df = dukascopy_python.fetch(
            instrument = INSTRUMENT_MAP[symbol],
            interval   = INTERVAL_MAP[timeframe],
            offer_side = dukascopy_python.OFFER_SIDE_BID,
            start      = start,
            end        = end,
        )

        df.columns = [c.capitalize() for c in df.columns]
        log.info(f"Fetched {len(df)} bars for {symbol}")
        return df

    def fetch_and_cache(self, symbol: str, timeframe: str,
                        start: datetime, end: datetime,
                        cache_path: str) -> pd.DataFrame:
        """Fetch once, save to parquet, load locally next time"""
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            log.info(f"Loading cached data from {cache_path}")
            return pd.read_parquet(cache_path)

        df = self.get_ohlcv(symbol, timeframe, start=start, end=end)
        df.to_parquet(cache_path)
        log.info(f"Cached data saved to {cache_path}")
        return df