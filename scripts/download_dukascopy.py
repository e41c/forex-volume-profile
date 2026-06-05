# scripts/download_dukascopy.py
"""
Download Dukascopy candles (WITH real tick volume) into DuckDB via dukascopy-node.

Why Dukascopy over HistData: HistData M1 has zero volume, so we had to fake it with a
saturating bar-count proxy that disabled the volume-spike filter (our biggest edge).
Dukascopy is a real ECN broker — its candle volume is genuine tick activity that spikes,
so the full volume strategy (profile weighting + 1.4× spike confirmation) works again.

Requires Node.js (uses `npx dukascopy-node`). Output is UTC, candles direct (no tick
aggregation), tiny (~7MB/pair for 20yr H1).

Usage:
    python scripts/download_dukascopy.py GBPUSD --from 2004-01-01
    python scripts/download_dukascopy.py EURUSD GBPUSD AUDUSD NZDUSD USDJPY USDCAD USDCHF
    python scripts/download_dukascopy.py --majors --from 2004-01-01

Each (symbol, timeframe) REPLACES any existing rows for that symbol+timeframe in DuckDB,
so re-running is safe and sources never mix.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import tempfile
import pandas as pd

from src.data.db_manager import get_connection, initialize_db, get_stats
from src.utils.logger import get_logger

log = get_logger("dukascopy")

# timeframes the strategy needs → dukascopy-node tokens
TIMEFRAMES = {"H1": "h1", "M15": "m15"}
MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF"]


def download_csv(symbol: str, tf_node: str, date_from: str, date_to: str,
                 outdir: str) -> str:
    """Invoke dukascopy-node; return the path to the written CSV."""
    inst = symbol.lower()
    fname = f"{inst}_{tf_node}"
    cmd = [
        "npx", "--yes", "dukascopy-node",
        "-i", inst, "-from", date_from, "-to", date_to,
        "-t", tf_node, "-v", "-vu", "units", "-f", "csv",
        "-r", "5", "-rp", "800", "-bp", "500",
        # resilience for long multi-year pulls: skip a single failed artifact after
        # retries instead of aborting. (Do NOT use -re/retry-on-empty here: weekends
        # and holidays are legitimately empty, and retrying them all makes M15 hang.)
        "-fr",
        "-dir", outdir, "-fn", fname, "-s",
    ]
    log.info(f"  {symbol} {tf_node}: downloading {date_from} → {date_to} ...")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    path = os.path.join(outdir, f"{fname}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dukascopy-node produced no file for {symbol} {tf_node}")
    return path


def load_csv(path: str, symbol: str, tf: str) -> pd.DataFrame:
    """Dukascopy CSV (timestamp ms, O,H,L,C,volume) → ohlcv-table schema."""
    raw = pd.read_csv(path)  # columns: timestamp,open,high,low,close,volume
    out = pd.DataFrame({
        "time":      pd.to_datetime(raw["timestamp"], unit="ms", utc=True),
        "symbol":    symbol.upper(),
        "timeframe": tf,
        "open":      raw["open"].astype(float),
        "high":      raw["high"].astype(float),
        "low":       raw["low"].astype(float),
        "close":     raw["close"].astype(float),
        "volume":    raw["volume"].round().astype("int64"),
    })
    out = out.drop_duplicates(subset=["time"]).sort_values("time")
    return out[["time", "symbol", "timeframe", "open", "high", "low", "close", "volume"]]


def ingest(symbol: str, date_from: str, date_to: str, con, only_tfs=None) -> None:
    tfs = {k: v for k, v in TIMEFRAMES.items() if (only_tfs is None or k in only_tfs)}
    with tempfile.TemporaryDirectory() as tmp:
        for tf, tf_node in tfs.items():
            path = download_csv(symbol, tf_node, date_from, date_to, tmp)
            df = load_csv(path, symbol, tf)
            if df.empty:
                log.warning(f"  {symbol} {tf}: empty — skipping")
                continue
            # REPLACE this symbol+timeframe so sources never mix
            con.execute("DELETE FROM ohlcv WHERE symbol=? AND timeframe=?",
                        [symbol.upper(), tf])
            con.execute("INSERT INTO ohlcv SELECT * FROM df")
            log.info(f"  {symbol} {tf}: {len(df):,} bars  "
                     f"{df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}  "
                     f"(volume mean {df['volume'].mean():,.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", help="symbols, e.g. EURUSD GBPUSD")
    ap.add_argument("--majors", action="store_true", help="the 7 majors (EURUSD..USDCHF)")
    ap.add_argument("--from", dest="date_from", default="2004-01-01")
    ap.add_argument("--to", dest="date_to", default="now")
    ap.add_argument("--timeframes", nargs="+", default=None,
                    help="subset of H1 M15 (default both)")
    args = ap.parse_args()

    symbols = MAJORS if args.majors else [s.upper() for s in args.symbols]
    if not symbols:
        log.error("Give symbols or --majors. e.g. python scripts/download_dukascopy.py GBPUSD")
        sys.exit(1)

    log.info("=" * 55)
    log.info("GOBLIN DUKASCOPY DOWNLOADER 🐲  (real volume!)")
    log.info("=" * 55)
    log.info(f"Symbols: {symbols}  |  {args.date_from} → {args.date_to}")

    initialize_db()
    con = get_connection()
    try:
        only = [t.upper() for t in args.timeframes] if args.timeframes else None
        for s in symbols:
            log.info(f"── {s} ──")
            ingest(s, args.date_from, args.date_to, con, only_tfs=only)
    finally:
        con.close()

    log.info("\n📊 Database summary:")
    print(get_stats().to_string(index=False))
