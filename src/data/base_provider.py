# src/data/base_provider.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseDataProvider(ABC):

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, **kwargs) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        ['Open', 'High', 'Low', 'Close', 'Volume']
        DatetimeIndex
        """
        pass