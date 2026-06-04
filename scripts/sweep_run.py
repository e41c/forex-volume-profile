# scripts/sweep_run.py
"""
Single parameter-sweep backtest run — one config variant, one metrics line.

Built for fanning out parallel experiments: each process monkeypatches Config
in-memory (no file edits, no cross-run interference), reads the DuckDB read-only
(many concurrent readers OK), and prints ONE machine-readable result line. It does
NOT save the trade journal/chart, so parallel runs never clobber each other.

Usage:
    python scripts/sweep_run.py --label baseline
    python scripts/sweep_run.py --label rr3 --set MAX_RR_RATIO=3.0
    python scripts/sweep_run.py --label no_va --set ENABLE_VA_EDGE_FADES=False
    python scripts/sweep_run.py --label vol14 --set VOLUME_SPIKE_MULT=1.4 --set MIN_RR_RATIO=1.8

Output line (parseable):
    RESULT label=<x> trades=<n> win=<%> pf=<x> net_pct=<x> dd=<x> sharpe=<x> ...
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from src.config import Config


def _coerce(value: str):
    """Turn a CLI string into bool / int / float / str."""
    if value in ("True", "False"):
        return value == "True"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(pairs: list[str]) -> dict:
    applied = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects NAME=VALUE, got {pair!r}")
        name, raw = pair.split("=", 1)
        name = name.strip()
        if not hasattr(Config, name):
            raise ValueError(f"Config has no attribute {name!r}")
        val = _coerce(raw.strip())
        setattr(Config, name, val)
        applied[name] = val
    return applied


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="Config override NAME=VALUE (repeatable)")
    args = ap.parse_args(argv)

    applied = apply_overrides(args.overrides)

    # Imports AFTER overrides so any module-load-time reads see new values.
    from src.data.csv_provider import CSVProvider
    from src.backtester import run_backtest, TradingCosts

    provider = CSVProvider()
    df = provider.get_ohlcv(symbol=Config.SYMBOL, timeframe=Config.TIMEFRAME_PROFILE)
    df_m15 = provider.get_ohlcv(symbol=Config.SYMBOL, timeframe=Config.TIMEFRAME_TREND)
    df_m30 = df_m15.resample("30min").agg({
        "Open": "first", "High": "max", "Low": "min",
        "Close": "last", "Volume": "sum",
    }).dropna()

    costs = TradingCosts(spread_pips=1.2, slippage_pips=0.5, commission=0.0,
                         swap_long=-0.8, swap_short=0.2)

    result = run_backtest(
        df=df, df_m15=df_m30, costs=costs,
        profile_window=Config.PROFILE_WINDOW,
        warmup_bars=Config.PROFILE_WINDOW,
        starting_balance=Config.ACCOUNT_BALANCE,
        risk_percent=Config.RISK_PERCENT,
        use_session_profiles=True,
        entry_wick_ratio=Config.ENTRY_WICK_RATIO,
        entry_min_body_pips=Config.ENTRY_MIN_BODY_PIPS,
        verbose=False,
    )

    print(
        f"RESULT label={args.label} "
        f"trades={result.total_trades} "
        f"win={result.win_rate} "
        f"pf={result.profit_factor} "
        f"net_pct={result.net_profit_pct:.1f} "
        f"dd={result.max_drawdown_pct:.1f} "
        f"sharpe={result.sharpe_ratio} "
        f"overrides={applied}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
