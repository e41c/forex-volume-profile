# src/data/mt5_provider.py — Windows only
import pandas as pd
from .base_provider import BaseDataProvider
from src.utils.logger import get_logger
from src.config import Config

log = get_logger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

TIMEFRAME_MAP = {
    "M1":  "TIMEFRAME_M1",
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1":  "TIMEFRAME_H1",
    "D1":  "TIMEFRAME_D1",
}

class MT5DataProvider(BaseDataProvider):

    def __init__(self):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not available — Windows only")

        if not mt5.initialize(
            login    = int(Config.MT5_LOGIN),
            password = Config.MT5_PASSWORD,
            server   = Config.MT5_SERVER
        ):
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

        log.info("MT5 connected successfully")

    def get_ohlcv(self, symbol: str, timeframe: str,
                  bars: int = 2000, **kwargs) -> pd.DataFrame:

        tf = getattr(mt5, TIMEFRAME_MAP[timeframe])
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = df.rename(columns={
            'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close',
            'tick_volume': 'Volume'
        })

        log.info(f"MT5: fetched {len(df)} bars for {symbol} {timeframe}")
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def shutdown(self):
        mt5.shutdown()
        log.info("MT5 disconnected")