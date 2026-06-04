# scripts/ingest_mt5.py
"""
Backfill full MT5 price history into DuckDB — the bridge between the live MT5
terminal and the offline backtester.

Best-practice workflow (run on the Windows box with the MT5 terminal open):

    python scripts/ingest_mt5.py EURUSD GBPUSD EURGBP AUDUSD USDCHF \
        --timeframes H1 M15 --from 2003-01-01

    → bars land in data/db/forex.db → then backtest OFFLINE, fast & deterministic:

    python scripts/run_multi_pair.py

Why a dedicated ingest script (and not MT5DataProvider.get_ohlcv):
    MT5DataProvider uses copy_rates_from_pos(0, N) — only the most RECENT N bars.
    Backfilling 20 years needs copy_rates_from(symbol, tf, cursor, chunk) paged
    backward through time until the terminal runs dry. We never backtest directly
    against MT5 (slow, rate-limited, history-capped) — we land it in DuckDB once,
    then read from DuckDB forever.

⚠️  TIMEZONE CAVEAT — read this before mixing MT5 and CSV data:
    The existing CSV pairs were carefully converted from broker GMT+2/+3 (US-DST)
    to true UTC (see csv_provider.broker_to_utc). MT5 copy_rates returns timestamps
    in the BROKER SERVER's timezone, which for most brokers is also GMT+2/+3.
    If you ingest MT5 bars as-is while CSV bars are UTC, the two sets are misaligned
    by 2-3 hours — which corrupts any shared-timeline / portfolio backtest.

    Use --server-offset-hours to subtract the broker's UTC offset so MT5 bars match
    the CSV convention. Most forex.com-style brokers: GMT+2 winter, GMT+3 summer.
    Since the offset shifts with DST, the safest path for HISTORICAL backfill is the
    CSV pipeline (which handles DST per-bar). Reserve MT5 ingest for pairs you can't
    get as CSV, or for topping up recent bars — and verify alignment on one overlap
    day before trusting a portfolio run.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.db_manager import get_connection, initialize_db, get_stats
from src.utils.logger import get_logger

log = get_logger("ingest_mt5")

# Chunk size per request. MT5 caps bars per call by the terminal's
# "Max bars in chart" setting; 100k is comfortably under the usual default.
CHUNK = 100_000


def _mt5():
    """Lazy MT5 import with a clear message off-Windows."""
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise RuntimeError(
            "MetaTrader5 package not available — ingest must run on Windows with "
            "the MT5 terminal installed. On Mac, get pair data as CSV instead and "
            "use scripts/import_csv.py."
        ) from e
    return mt5


def fetch_full_history(mt5, symbol: str, tf_name: str,
                       date_from: datetime) -> pd.DataFrame:
    """
    Page backward from now to date_from, accumulating all available bars.

    Returns an empty DataFrame if the symbol/timeframe has no data.
    """
    tf_const = getattr(mt5, f"TIMEFRAME_{tf_name}", None)
    if tf_const is None:
        raise ValueError(f"Unknown timeframe {tf_name!r} — MT5 has no TIMEFRAME_{tf_name}")

    cursor = datetime.now(timezone.utc)
    frames = []

    while True:
        rates = mt5.copy_rates_from(symbol, tf_const, cursor, CHUNK)
        if rates is None or len(rates) == 0:
            break

        chunk_df = pd.DataFrame(rates)
        earliest = pd.to_datetime(chunk_df['time'].min(), unit='s', utc=True)
        latest   = pd.to_datetime(chunk_df['time'].max(), unit='s', utc=True)
        frames.append(chunk_df)
        log.info(f"  {symbol} {tf_name}: +{len(chunk_df):>7,} bars  "
                 f"({earliest.date()} → {latest.date()})")

        # Reached the requested start, or the start of available history.
        if earliest <= pd.Timestamp(date_from) or len(rates) < CHUNK:
            break

        # Step the cursor just before the earliest bar we got, and page again.
        cursor = earliest.to_pydatetime() - timedelta(seconds=1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['time']).sort_values('time')
    return df


def normalize_for_db(df: pd.DataFrame, symbol: str, tf_name: str,
                     server_offset_hours: float) -> pd.DataFrame:
    """
    Shape MT5 rates to the ohlcv table schema:
        time(TIMESTAMPTZ), symbol, timeframe, open, high, low, close, volume(BIGINT)

    server_offset_hours is SUBTRACTED to convert broker server time → UTC so the
    bars line up with the CSV-imported pairs.
    """
    out = pd.DataFrame()
    t = pd.to_datetime(df['time'], unit='s', utc=True)
    if server_offset_hours:
        t = t - pd.Timedelta(hours=server_offset_hours)

    out['time']      = t
    out['symbol']    = symbol.upper()
    out['timeframe'] = tf_name.upper()
    out['open']      = df['open'].astype(float)
    out['high']      = df['high'].astype(float)
    out['low']       = df['low'].astype(float)
    out['close']     = df['close'].astype(float)
    # MT5 gives tick_volume (count of ticks) — same proxy the CSV pipeline uses.
    out['volume']    = df['tick_volume'].astype('int64')

    return out[['time', 'symbol', 'timeframe',
                'open', 'high', 'low', 'close', 'volume']]


def ingest(symbols: list[str], timeframes: list[str],
           date_from: datetime, server_offset_hours: float) -> None:
    mt5 = _mt5()

    from src.config import Config
    if not mt5.initialize(
        login    = int(Config.MT5_LOGIN) if Config.MT5_LOGIN else 0,
        password = Config.MT5_PASSWORD or "",
        server   = Config.MT5_SERVER or "",
    ):
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    log.info("MT5 connected")

    initialize_db()
    con = get_connection()

    try:
        for symbol in symbols:
            for tf_name in timeframes:
                log.info(f"Fetching {symbol} {tf_name} from {date_from.date()}...")
                raw = fetch_full_history(mt5, symbol, tf_name, date_from)
                if raw.empty:
                    log.warning(f"  {symbol} {tf_name}: no data returned — skipping")
                    continue

                db_df = normalize_for_db(raw, symbol, tf_name, server_offset_hours)
                db_df = db_df[db_df['time'] >= pd.Timestamp(date_from)]

                before = con.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
                    [symbol.upper(), tf_name.upper()]
                ).fetchone()[0]

                # INSERT OR IGNORE on PK (time, symbol, timeframe) — safe to re-run.
                con.execute("INSERT OR IGNORE INTO ohlcv SELECT * FROM db_df")

                after = con.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
                    [symbol.upper(), tf_name.upper()]
                ).fetchone()[0]

                log.info(f"  {symbol} {tf_name}: +{after - before:,} new rows "
                         f"({after:,} total in DB)")
    finally:
        con.close()
        mt5.shutdown()
        log.info("MT5 disconnected")

    log.info("\n📊 Database summary:")
    stats = get_stats()
    if not stats.empty:
        print(stats.to_string(index=False))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill MT5 price history into DuckDB (Windows + MT5 terminal)."
    )
    p.add_argument("symbols", nargs="+",
                   help="Symbols to fetch, e.g. EURUSD GBPUSD EURGBP")
    p.add_argument("--timeframes", nargs="+", default=["H1", "M15"],
                   help="Timeframes to fetch (default: H1 M15 — what the strategy needs)")
    p.add_argument("--from", dest="date_from", default="2003-01-01",
                   help="Earliest date to fetch, YYYY-MM-DD (default: 2003-01-01)")
    p.add_argument("--server-offset-hours", type=float, default=0.0,
                   help="Hours to SUBTRACT from MT5 server time to reach UTC, so bars "
                        "align with CSV-imported pairs (e.g. 2 or 3 for GMT+2/+3). "
                        "Default 0 — see the timezone caveat in this file's docstring.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    date_from = datetime.strptime(args.date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    log.info("=" * 55)
    log.info("GOBLIN MT5 → DUCKDB INGEST 🐲")
    log.info("=" * 55)
    log.info(f"Symbols:    {args.symbols}")
    log.info(f"Timeframes: {args.timeframes}")
    log.info(f"From:       {date_from.date()}")
    if args.server_offset_hours:
        log.info(f"UTC shift:  −{args.server_offset_hours}h (server → UTC)")
    else:
        log.warning("No --server-offset-hours given: MT5 times stored as-is. "
                    "Verify alignment before mixing with CSV pairs in a portfolio run.")

    ingest(args.symbols, args.timeframes, date_from, args.server_offset_hours)
