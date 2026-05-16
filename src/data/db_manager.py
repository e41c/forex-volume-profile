# src/data/db_manager.py
import duckdb
import os
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

def get_connection() -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(Config.DATA_DB), exist_ok=True)
    return duckdb.connect(Config.DATA_DB)

def initialize_db():
    """Create tables and indexes if they don't exist"""
    con = get_connection()
    con.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            time        TIMESTAMPTZ  NOT NULL,
            symbol      VARCHAR(20)  NOT NULL,
            timeframe   VARCHAR(10)  NOT NULL,
            open        DOUBLE       NOT NULL,
            high        DOUBLE       NOT NULL,
            low         DOUBLE       NOT NULL,
            close       DOUBLE       NOT NULL,
            volume      BIGINT       NOT NULL,
            PRIMARY KEY (time, symbol, timeframe)
        )
    """)

    # fast lookup index — symbol + timeframe + time range
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_main
        ON ohlcv (symbol, timeframe, time)
    """)
    con.close()
    log.info("DuckDB initialized successfully")

def get_stats() -> dict:
    """Return row counts per symbol/timeframe"""
    con = get_connection()
    result = con.execute("""
        SELECT
            symbol,
            timeframe,
            COUNT(*)            AS bars,
            MIN(time)           AS earliest,
            MAX(time)           AS latest
        FROM ohlcv
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """).df()
    con.close()
    return result