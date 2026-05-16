# src/data/csv_provider.py
import os
import re
import glob
import numpy as np
import pandas as pd
import pytz
import duckdb
from datetime import datetime, timezone
from src.data.base_provider import BaseDataProvider
from src.data.db_manager import get_connection, initialize_db
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

US_EASTERN = pytz.timezone('US/Eastern')

# matches: EURUSD_GMT+2_US-DST_2003_bars_M1.csv
FILENAME_PATTERN = re.compile(
    r'(?P<symbol>[A-Z]+)_GMT\+2_US-DST_(?P<year>\d{4})_bars_(?P<timeframe>M1|M5|M15|M30|H1|H4|D1|W1|MN1)\.csv',
    re.IGNORECASE
)


def broker_to_utc(dt_series: pd.Series) -> pd.Series:
    """
    Broker time is GMT+2 base, shifts to GMT+3 during US DST.
    US DST = second Sunday March to first Sunday November.
    We subtract 2h or 3h to get UTC.
    """
    def get_offset(dt: pd.Timestamp) -> pd.Timedelta:
        try:
            eastern_dt = US_EASTERN.localize(
                dt.replace(hour=12, minute=0, second=0,
                           microsecond=0, tzinfo=None)
            )
            in_dst = bool(eastern_dt.dst().total_seconds() > 0)
            return pd.Timedelta(hours=3 if in_dst else 2)
        except Exception:
            return pd.Timedelta(hours=2)  # safe fallback

    offsets = dt_series.apply(get_offset)
    return dt_series - offsets


def parse_filename(filepath: str) -> dict | None:
    """Extract symbol, year, timeframe from filename"""
    name  = os.path.basename(filepath)
    match = FILENAME_PATTERN.match(name)
    if not match:
        log.warning(f"Skipping unrecognised filename: {name}")
        return None
    return match.groupdict()


def load_csv_to_df(filepath: str, symbol: str,
                   timeframe: str) -> pd.DataFrame:
    """
    Load one broker CSV file.
    Format: date,time,open,high,low,close,volume  (no header row)
    Date:   YYYY.MM.DD   Time: HH:MM:SS
    """
    df = pd.read_csv(
        filepath,
        header=None,
        names=['date', 'time_str', 'open',
               'high', 'low', 'close', 'volume'],
        dtype={
            'date':     str,
            'time_str': str,
            'open':     np.float64,
            'high':     np.float64,
            'low':      np.float64,
            'close':    np.float64,
            'volume':   np.int64,
        }
    )

    # parse broker datetime
    df['broker_dt'] = pd.to_datetime(
        df['date'] + ' ' + df['time_str'],
        format='%Y.%m.%d %H:%M:%S'
    )

    # convert broker GMT+2/+3 US DST -> UTC
    df['time'] = broker_to_utc(df['broker_dt'])
    df['time'] = df['time'].dt.tz_localize('UTC')

    df['symbol']    = symbol.upper()
    df['timeframe'] = timeframe.upper()

    df = df[['time', 'symbol', 'timeframe',
             'open', 'high', 'low', 'close', 'volume']]
    df = df.drop_duplicates(subset=['time']).sort_values('time')

    return df


def already_imported(symbol: str, timeframe: str, year: int) -> bool:
    """Check if a specific year/symbol/timeframe is already in DB"""
    con    = get_connection()
    result = con.execute("""
        SELECT COUNT(*) FROM ohlcv
        WHERE symbol    = ?
          AND timeframe = ?
          AND YEAR(time) = ?
    """, [symbol.upper(), timeframe.upper(), year]).fetchone()
    con.close()
    return result[0] > 0


def import_all_csvs(raw_dir: str = Config.DATA_RAW,
                    skip_existing: bool = True):
    """
    Walk raw_dir recursively, import every recognised CSV file.
    Skips years already in the DB unless skip_existing=False.
    """
    initialize_db()

    # find all CSVs recursively
    all_files = glob.glob(
        os.path.join(raw_dir, '**', '*.csv'), recursive=True
    )

    if not all_files:
        log.warning(f"No CSV files found under {raw_dir}")
        return

    log.info(f"Found {len(all_files)} CSV files")
    con = get_connection()

    imported = 0
    skipped  = 0
    failed   = 0

    for filepath in sorted(all_files):
        meta = parse_filename(filepath)
        if meta is None:
            failed += 1
            continue

        symbol    = meta['symbol']
        timeframe = meta['timeframe']
        year      = int(meta['year'])

        if skip_existing and already_imported(symbol, timeframe, year):
            log.info(f"Skipping {os.path.basename(filepath)} — already in DB")
            skipped += 1
            continue

        try:
            df = load_csv_to_df(filepath, symbol, timeframe)
            con.execute("""
                INSERT OR IGNORE INTO ohlcv
                SELECT * FROM df
            """)
            log.info(
                f"Imported {len(df):>8,} bars  "
                f"{symbol} {timeframe} {year}  "
                f"[{os.path.basename(filepath)}]"
            )
            imported += 1

        except Exception as e:
            log.error(f"Failed to import {filepath}: {e}")
            failed += 1

    con.close()
    log.info(
        f"Import complete — "
        f"{imported} imported, {skipped} skipped, {failed} failed"
    )


class CSVProvider(BaseDataProvider):
    """
    Reads OHLCV data from DuckDB.
    Returns UTC-indexed DataFrame in MT5-compatible format.
    """

    def get_ohlcv(self,
                  symbol:    str,
                  timeframe: str,
                  start:     datetime = None,
                  end:       datetime = None,
                  bars:      int      = None,
                  **kwargs) -> pd.DataFrame:

        con    = get_connection()
        params = [symbol.upper(), timeframe.upper()]
        where  = "WHERE symbol = ? AND timeframe = ?"

        if start:
            where += " AND time >= ?"
            params.append(start)
        if end:
            where += " AND time <= ?"
            params.append(end)

        limit = f"LIMIT {bars}" if bars and not (start or end) else ""

        query = f"""
            SELECT time, open, high, low, close, volume
            FROM ohlcv
            {where}
            ORDER BY time
            {limit}
        """

        df = con.execute(query, params).df()

        # if date range returned nothing, load all available data instead
        if df.empty and (start or end):
            log.warning(
                f"No data found for {symbol} {timeframe} in requested "
                f"date range — loading all available data instead"
            )
            df = con.execute("""
                SELECT time, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time
            """, [symbol.upper(), timeframe.upper()]).df()

        con.close()

        if df.empty:
            raise ValueError(
                f"No data found for {symbol} {timeframe} in database.\n"
                f"Run: python scripts/import_csv.py"
            )

        # normalize to MT5 column format
        df = df.set_index('time')
        df.index.name = 'time'
        df.columns    = ['Open', 'High', 'Low', 'Close', 'Volume']

        log.info(
            f"Loaded {len(df):,} bars  {symbol} {timeframe}  "
            f"{df.index[0].date()} -> {df.index[-1].date()}"
        )
        return df
