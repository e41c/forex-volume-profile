# scripts/import_histdata.py
"""
Import HistData.com M1 CSVs into DuckDB as H1 + M15 bars.

HistData generic-ASCII format (DAT_MT_<SYMBOL>_M1_<YEAR>.csv):
    YYYY.MM.DD,HH:MM,open,high,low,close,volume
  - volume is ALWAYS 0 (HistData free data carries no volume)
  - a bar is only written for minutes where price moved (gaps = no activity)
  - timestamps are US Eastern wall-clock

Because there is no real volume, we use the **count of M1 bars per period** as the
volume proxy — i.e. how much price activity occurred. This is the Market-Profile / TPO
idea (time-at-price), and it drives both the volume profile and the volume-confirmation
filter. NOTE: the legacy EURUSD data uses real tick volume, so the two are not a perfect
apples-to-apples basis.

Timestamps are localized as US/Eastern (the same tz the session filter uses) then stored
as UTC, so the London/NY session gating sees the original wall-clock times.

Usage:
    python scripts/import_histdata.py GBPUSD              # one pair
    python scripts/import_histdata.py GBPUSD AUDUSD USDJPY
    python scripts/import_histdata.py --all               # every pair folder with HistData files
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import argparse
import numpy as np
import pandas as pd

from src.data.db_manager import get_connection, initialize_db, get_stats
from src.config import Config
from src.utils.logger import get_logger

log = get_logger("import_histdata")

HISTDATA_GLOB = "DAT_MT_*_M1_*.csv"
FNAME_RE = re.compile(r"DAT_MT_(?P<symbol>[A-Z]+)_M1_(?P<year>\d{4})\.csv", re.IGNORECASE)

# M1 → these timeframes (what the strategy needs)
RESAMPLE = {"H1": "1h", "M15": "15min"}


def load_m1_file(path: str) -> pd.DataFrame:
    """Read one HistData M1 csv → UTC-indexed OHLC DataFrame (volume dropped, it's 0)."""
    df = pd.read_csv(
        path, header=None,
        names=["date", "tstr", "Open", "High", "Low", "Close", "_vol"],
        dtype={"date": str, "tstr": str,
               "Open": np.float64, "High": np.float64,
               "Low": np.float64, "Close": np.float64, "_vol": np.float64},
    )
    naive = pd.to_datetime(df["date"] + " " + df["tstr"], format="%Y.%m.%d %H:%M")
    # localize US/Eastern; DST gaps/overlaps are rare M1 edge cases — drop them
    et = naive.dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="NaT")
    df = df.assign(time=et.dt.tz_convert("UTC")).dropna(subset=["time"])
    return df.set_index("time")[["Open", "High", "Low", "Close"]].sort_index()


def resample_with_volume_proxy(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate M1 → target timeframe. Volume = count of M1 bars in the period (proxy)."""
    out = m1.resample(rule).agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"),
        Volume=("Open", "count"),
    ).dropna(subset=["Open"])
    out = out[out["Volume"] > 0]                 # drop empty periods (weekends/gaps)
    out["Volume"] = out["Volume"].astype("int64")
    return out


def import_symbol(symbol: str, raw_dir: str, con) -> None:
    files = sorted(glob.glob(os.path.join(raw_dir, symbol, HISTDATA_GLOB)))
    files = [f for f in files if FNAME_RE.search(os.path.basename(f))]
    if not files:
        log.warning(f"{symbol}: no HistData M1 files found — skipping")
        return

    log.info(f"{symbol}: loading {len(files)} M1 year-file(s)...")
    m1 = pd.concat([load_m1_file(f) for f in files])
    m1 = m1[~m1.index.duplicated(keep="first")].sort_index()
    log.info(f"{symbol}: {len(m1):,} M1 bars  {m1.index[0].date()} → {m1.index[-1].date()}")

    for tf, rule in RESAMPLE.items():
        bars = resample_with_volume_proxy(m1, rule)
        db = bars.reset_index().rename(columns=str.lower)
        db["symbol"] = symbol.upper()
        db["timeframe"] = tf
        db = db[["time", "symbol", "timeframe", "open", "high", "low", "close", "volume"]]

        before = con.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
                             [symbol.upper(), tf]).fetchone()[0]
        con.execute("INSERT OR IGNORE INTO ohlcv SELECT * FROM db")
        after = con.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
                            [symbol.upper(), tf]).fetchone()[0]
        log.info(f"  {symbol} {tf}: +{after - before:,} rows  "
                 f"(volume proxy: mean {bars['Volume'].mean():.0f} bars/period)")


def discover_symbols(raw_dir: str) -> list[str]:
    syms = set()
    for f in glob.glob(os.path.join(raw_dir, "*", HISTDATA_GLOB)):
        m = FNAME_RE.search(os.path.basename(f))
        if m:
            syms.add(m.group("symbol").upper())
    return sorted(syms)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="symbols to import (e.g. GBPUSD AUDUSD)")
    ap.add_argument("--all", action="store_true", help="import every pair with HistData files")
    ap.add_argument("--raw-dir", default=Config.DATA_RAW)
    args = ap.parse_args()

    log.info("=" * 55)
    log.info("GOBLIN HISTDATA IMPORTER 🐲")
    log.info("=" * 55)

    symbols = discover_symbols(args.raw_dir) if args.all else [s.upper() for s in args.symbols]
    if not symbols:
        log.error("No symbols given. Use: python scripts/import_histdata.py GBPUSD  (or --all)")
        sys.exit(1)

    log.info(f"Importing: {symbols}")
    initialize_db()
    con = get_connection()
    try:
        for s in symbols:
            import_symbol(s, args.raw_dir, con)
    finally:
        con.close()

    log.info("\n📊 Database summary:")
    print(get_stats().to_string(index=False))
