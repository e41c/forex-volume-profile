# scripts/run_multi_pair.py
"""
Multi-pair walk-forward backtest.

Runs the same VP mean-reversion strategy across all pairs that have data
in the DuckDB database, then prints a ranked comparison table.

Usage:
    python scripts/run_multi_pair.py              # all pairs in DB
    python scripts/run_multi_pair.py GBPUSD USDJPY  # specific pairs

Pip value reference (CAD per pip per 1.0 standard lot, approx rates):
    xxxUSD (EURUSD, GBPUSD, AUDUSD, NZDUSD):
        pip_val_usd = 100000 * 0.0001 = $10 USD  → $10 * USDCAD
    USDJPY:
        pip_val_usd = 100000 * 0.01 / 150 ≈ $6.67 USD → * USDCAD
    USDCAD:
        pip_val_cad = 100000 * 0.0001 / USDCAD ≈ $7.35 CAD  (quote IS CAD)
    USDCHF:
        pip_val_usd = 100000 * 0.0001 / 0.90 ≈ $11.11 USD → * USDCAD
    EURGBP:
        pip_val_gbp = 100000 * 0.0001 = £10 GBP → * GBPCAD ≈ * 1.73

Update pip_value_cad_per_lot when rates move significantly (>5%).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import pandas as pd
from dataclasses import dataclass
from src.data import get_provider
from src.data.base_provider import BaseDataProvider
from src.backtester import run_backtest, print_results, save_trade_journal, TradingCosts
from src.config import Config
from src.utils.logger import get_logger

log = get_logger("multi_pair")

# ── Per-pair cost and pip-value presets ──────────────────────────────────────
# Update these when market rates shift significantly.
# All pip values in CAD per pip per 1.0 standard lot.
# Spreads are approximate forex.com standard account averages.

USDCAD_RATE = 1.36   # update to current rate

@dataclass
class PairPreset:
    symbol:               str
    pip_size:             float   # 0.0001 for most, 0.01 for JPY
    spread_pips:          float   # forex.com standard avg spread
    swap_long:            float   # pips/night
    swap_short:           float   # pips/night
    pip_value_cad_per_lot: float  # CAD per pip per 1.0 standard lot
    note:                 str = ""


PAIR_PRESETS = {
    # ── Majors (xxxUSD) — highest liquidity, tightest spreads ───────────────
    # pip_value_usd = 10, × USDCAD 1.36
    "EURUSD": PairPreset("EURUSD", 0.0001, 1.2, -0.8,  0.2,  13.60, "benchmark"),
    "GBPUSD": PairPreset("GBPUSD", 0.0001, 1.5, -1.2,  0.4,  13.60, "similar to EUR, wider spread"),
    "AUDUSD": PairPreset("AUDUSD", 0.0001, 1.6, -0.5, -0.2,  13.60, "commodity proxy, often ranges"),
    "NZDUSD": PairPreset("NZDUSD", 0.0001, 2.0, -0.3, -0.3,  13.60, "thin, AUD correlation"),

    # ── USDxxx pairs ─────────────────────────────────────────────────────────
    # USDJPY: pip_val_usd = 100000 * 0.01 / 150 ≈ $6.67 → × 1.36 ≈ $9.07 CAD
    "USDJPY": PairPreset("USDJPY", 0.01,   1.5,  0.5, -2.0,   9.07, "safe-haven, JPY pip=0.01"),

    # USDCAD: quote IS CAD, pip_val = 100000 * 0.0001 / USDCAD ≈ $7.35 CAD
    "USDCAD": PairPreset("USDCAD", 0.0001, 2.0, -1.5, -0.5,   7.35, "oil-correlated"),

    # USDCHF: pip_val_usd = 100000 * 0.0001 / 0.90 ≈ $11.11 → × 1.36 ≈ $15.11 CAD
    "USDCHF": PairPreset("USDCHF", 0.0001, 1.8, -0.3, -1.0,  15.11, "EUR inverse, wide swap"),

    # ── Cross pairs ──────────────────────────────────────────────────────────
    # EURGBP: pip_val_gbp = £10, × GBPCAD ≈ 1.73 ≈ $17.30 CAD
    "EURGBP": PairPreset("EURGBP", 0.0001, 1.5, -0.7,  0.1,  17.30, "ranges well — dark horse"),

    # EURJPY: pip_val_jpy = 1000 JPY / 150 = $6.67 USD → × 1.36 ≈ $9.07 CAD
    "EURJPY": PairPreset("EURJPY", 0.01,   2.0, -0.5, -1.5,   9.07, "high volatility cross"),

    # GBPJPY: similar pip value to EURJPY, very volatile
    "GBPJPY": PairPreset("GBPJPY", 0.01,   2.5, -1.0, -1.5,   9.07, "very volatile, wide spread"),
}


def build_costs(preset: PairPreset) -> TradingCosts:
    return TradingCosts(
        spread_pips            = preset.spread_pips,
        commission             = 0.0,
        swap_long              = preset.swap_long,
        swap_short             = preset.swap_short,
        slippage_pips          = 0.5,
        pip_size               = preset.pip_size,
        pip_value_cad_per_lot  = preset.pip_value_cad_per_lot,
    )


def run_pair(symbol: str, provider: BaseDataProvider, h1_only: bool = False) -> dict | None:
    """
    Load data and run a full backtest for one pair.
    Returns a results summary dict, or None if data is missing.

    h1_only: ignore M15 and use H1 rejection candles for entries. Useful for a fast,
             consistent first read across pairs when M15 isn't downloaded yet.
    """
    preset = PAIR_PRESETS.get(symbol)
    if preset is None:
        log.warning(f"{symbol}: no preset config — add it to PAIR_PRESETS in this file")
        return None

    try:
        df = provider.get_ohlcv(symbol=symbol, timeframe="H1")
    except ValueError as e:
        log.warning(f"{symbol}: no H1 data in DB — {e}")
        return None

    df_m15 = None
    if h1_only:
        log.info(f"{symbol}: H1-only mode — ignoring M15, using H1 entry candles")
    else:
        try:
            df_m15 = provider.get_ohlcv(symbol=symbol, timeframe="M15")
        except ValueError:
            log.warning(f"{symbol}: no M15 data — falling back to H1-only entries")
            df_m15 = None

    df_m30 = None
    if df_m15 is not None:
        df_m30 = df_m15.resample('30min').agg({
            'Open':   'first',
            'High':   'max',
            'Low':    'min',
            'Close':  'last',
            'Volume': 'sum',
        }).dropna()

    costs  = build_costs(preset)
    t0     = time.time()

    log.info(f"{'─'*55}")
    log.info(f"Running {symbol}  ({preset.note})")
    log.info(f"{'─'*55}")

    result = run_backtest(
        df                   = df,
        df_m15               = df_m30,
        costs                = costs,
        pip_size             = preset.pip_size,
        profile_window       = Config.PROFILE_WINDOW,
        warmup_bars          = Config.PROFILE_WINDOW,
        starting_balance     = Config.ACCOUNT_BALANCE,
        risk_percent         = Config.RISK_PERCENT,
        use_session_profiles = True,
        entry_wick_ratio     = Config.ENTRY_WICK_RATIO,
        # JPY pairs (pip 0.01) have ~10× smaller pip-denominated bodies → scaled floor
        entry_min_body_pips  = Config.ENTRY_MIN_BODY_PIPS if preset.pip_size == 0.0001 else 0.15,
        verbose              = False,
    )

    elapsed = round(time.time() - t0, 1)
    log.info(
        f"{symbol}  done in {elapsed}s  |  "
        f"{result.total_trades} trades  "
        f"{result.win_rate}% win  "
        f"PF {result.profit_factor}  "
        f"{result.net_profit_pct:+.1f}%  "
        f"DD {result.max_drawdown_pct:.1f}%"
    )

    return {
        "symbol":    symbol,
        "trades":    result.total_trades,
        "win_pct":   result.win_rate,
        "pf":        result.profit_factor,
        "net_pct":   result.net_profit_pct,
        "dd_pct":    result.max_drawdown_pct,
        "sharpe":    result.sharpe_ratio,
        "net_cad":   result.net_profit,
        "elapsed_s": elapsed,
        "note":      preset.note,
        "_result":   result,
        "_costs":    costs,
    }


def print_summary(rows: list[dict]):
    """Print ranked comparison table."""
    if not rows:
        log.warning("No pairs completed successfully")
        return

    viable = [r for r in rows if r["pf"] >= 1.5 and r["dd_pct"] < 20]
    rows_sorted = sorted(rows, key=lambda r: r["pf"], reverse=True)

    print("\n" + "=" * 90)
    print(f"  MULTI-PAIR RESULTS  ({len(rows)} pairs tested, {len(viable)} viable)")
    print("=" * 90)
    print(f"  {'Symbol':<10} {'Trades':>7} {'Win%':>6} {'PF':>5} {'Net%':>7} {'DD%':>6} {'Sharpe':>7}  {'Status'}")
    print("  " + "─" * 88)

    for r in rows_sorted:
        pf_ok = r["pf"] >= 1.5
        dd_ok = r["dd_pct"] < 20
        wr_ok = r["win_pct"] >= 50.0
        all_ok = pf_ok and dd_ok

        status = "PASS" if all_ok else (
            "PF FAIL" if not pf_ok else
            "DD FAIL" if not dd_ok else
            "MARGINAL"
        )
        flag = "  *" if all_ok else ""

        print(
            f"  {r['symbol']:<10} "
            f"{r['trades']:>7} "
            f"{r['win_pct']:>6.1f} "
            f"{r['pf']:>5.2f} "
            f"{r['net_pct']:>+7.1f} "
            f"{r['dd_pct']:>6.1f} "
            f"{r['sharpe']:>7.2f}  "
            f"{status}{flag}"
        )

    if viable:
        combined_trades = sum(r["trades"] for r in viable)
        combined_cad    = sum(r["net_cad"] for r in viable)
        print("  " + "─" * 88)
        print(f"  {'COMBINED (viable)':10} {combined_trades:>7}  trades → "
              f"${combined_cad:+.0f} CAD net  "
              f"({combined_trades / 23:.1f} trades/year across {len(viable)} pairs)")

    print("=" * 90)
    print("  * = PASS (PF >= 1.5 and DD < 20%)")
    print()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Goblin multi-pair backtest")
    parser.add_argument(
        "symbols", nargs="*",
        help="Pairs to run (e.g. GBPUSD EURGBP). Default: all presets with DB data.",
    )
    parser.add_argument(
        "--source", choices=["offline", "mt5"], default="offline",
        help="Data source (default: offline — DuckDB. Backtests should run offline.)",
    )
    parser.add_argument(
        "--h1-only", action="store_true",
        help="Use H1 entry candles, ignore M15 (fast consistent read before M15 is loaded)",
    )
    parser.add_argument(
        "--asian", action="store_true",
        help="Also trade the Asian session (for AUD/NZD/JPY whose volume peaks in Tokyo)",
    )
    args = parser.parse_args()

    if args.asian:
        Config.INCLUDE_ASIAN_SESSION = True

    # parse args — optional list of symbols, else use all with data in DB
    requested = [s.upper() for s in args.symbols] if args.symbols else None

    log.info("=" * 55)
    log.info("GOBLIN MULTI-PAIR BACKTESTER")
    log.info("=" * 55)

    provider = get_provider(args.source)

    # discover what pairs are available in DB
    from src.data.db_manager import get_stats
    stats = get_stats()
    if stats.empty:
        log.error("No data in database. Run: python scripts/import_csv.py")
        sys.exit(1)

    available_h1 = set(
        stats[stats['timeframe'] == 'H1']['symbol'].str.upper().tolist()
    )

    if requested:
        pairs_to_run = [s for s in requested if s in available_h1]
        missing = [s for s in requested if s not in available_h1]
        if missing:
            log.warning(f"No H1 data for: {missing} — skipping")
    else:
        # run all pairs that have a preset config AND data in DB
        pairs_to_run = [s for s in PAIR_PRESETS if s in available_h1]

    if not pairs_to_run:
        log.error(
            f"No matching pairs found.\n"
            f"Pairs with H1 data: {sorted(available_h1)}\n"
            f"Pairs with presets: {sorted(PAIR_PRESETS.keys())}"
        )
        sys.exit(1)

    log.info(f"Running {len(pairs_to_run)} pairs: {pairs_to_run}")

    if args.h1_only:
        log.info("H1-only mode: entries on H1 candles, M15 ignored")

    results = []
    for symbol in pairs_to_run:
        row = run_pair(symbol, provider, h1_only=args.h1_only)
        if row:
            results.append(row)

    print_summary(results)

    # save charts for all pairs that produced trades
    try:
        from scripts.run_backtest import plot_equity_curve
        for r in results:
            if r["trades"] > 0:
                try:
                    plot_equity_curve(r["_result"], r["_costs"], symbol=r["symbol"])
                except Exception as e:
                    log.warning(f"Chart failed for {r['symbol']}: {e}")
    except ImportError:
        log.warning("Could not import plot_equity_curve — skipping charts")
