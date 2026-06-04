# src/data/__init__.py
"""
Data-provider factory.

One switch, two transports — pick the source explicitly instead of guessing
from the OS. The rest of the codebase depends only on the BaseDataProvider
contract (get_ohlcv(symbol, timeframe) -> OHLCV DataFrame), so backtester,
strategy, and main never need to know where the bars came from.

Usage:
    from src.data import get_provider

    provider = get_provider()            # honors Config.DATA_SOURCE / $DATA_SOURCE
    provider = get_provider("offline")   # explicit override (CSV/DuckDB)
    provider = get_provider("mt5")       # explicit override (live MetaTrader5)

Why a factory and not platform.system():
    "Where am I running" (Mac/Windows) and "what data do I want" (offline/live)
    are different questions. The Windows desktop wants BOTH offline backtests
    and live MT5 — so the choice has to be a knob, not a side effect of the OS.

    The MT5 import stays lazy (only imported inside the "mt5" branch), so this
    module imports cleanly on macOS where the MetaTrader5 package can't install.
"""
from src.config import Config
from src.data.base_provider import BaseDataProvider
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_provider(source: str | None = None) -> BaseDataProvider:
    """
    Return a data provider for the requested source.

    source: "offline" (CSV/DuckDB) or "mt5" (live MetaTrader5).
            Falls back to Config.DATA_SOURCE (env DATA_SOURCE) when None.
    """
    source = (source or Config.DATA_SOURCE).lower()

    if source in ("offline", "csv", "duckdb"):
        from src.data.csv_provider import CSVProvider
        log.info("Data source: offline (CSV/DuckDB)")
        return CSVProvider()

    if source in ("mt5", "live", "metatrader"):
        # Lazy import — MetaTrader5 is Windows-only and isn't installable on Mac.
        from src.data.mt5_provider import MT5DataProvider
        log.info("Data source: MT5 (live MetaTrader5)")
        return MT5DataProvider()

    raise ValueError(
        f"Unknown DATA_SOURCE {source!r} — use 'offline' (CSV/DuckDB) or 'mt5' (live)."
    )


__all__ = ["get_provider", "BaseDataProvider"]
