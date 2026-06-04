# src/main.py
import os
import argparse
from datetime import datetime, timezone
from src.config import Config
from src.data import get_provider
from src.indicators.volume_profile import build_volume_profile
from src.indicators.session_profile import build_multi_session_levels
from src.strategy.vp_strategy import generate_signal, calculate_position_size
from src.utils.logger import get_logger

log = get_logger(__name__)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("logs",           exist_ok=True)


def run(source: str = None):
    log.info("=== Goblin Bot Starting 🐲 ===")
    # source: "offline" | "mt5" | None (None → Config.DATA_SOURCE / $DATA_SOURCE)
    provider = get_provider(source)

    # --- Fetch Data ---
    df = provider.get_ohlcv(
        symbol    = Config.SYMBOL,
        timeframe = Config.TIMEFRAME_PROFILE,
    )

    # --- Build Long-Term Volume Profile (23 years) ---
    long_term_levels = build_volume_profile(df)
    log.info(
        f"Long-term  POC: {long_term_levels.poc:.5f}  |  "
        f"VAH: {long_term_levels.vah:.5f}  |  "
        f"VAL: {long_term_levels.val:.5f}  |  "
        f"HVNs: {len(long_term_levels.hvns)}  |  "
        f"LVNs: {len(long_term_levels.lvns)}"
    )

    # --- Priority 4: Build Multi-Session Profiles ---
    multi_levels = build_multi_session_levels(df)

    # --- Visualize (long-term profile) ---
    from src.visualizer import plot_volume_profile
    plot_volume_profile(df, long_term_levels, symbol=Config.SYMBOL)

    # --- Generate Signal (all 5 priorities active) ---
    signal = generate_signal(
        df           = df,
        levels       = long_term_levels,
        multi_levels = multi_levels,     # Priority 4 passed in
    )

    if signal:
        lots = calculate_position_size(
            account_balance = Config.ACCOUNT_BALANCE,
            risk_percent    = Config.RISK_PERCENT,
            stop_loss_pips  = abs(signal.entry - signal.stop_loss) / 0.0001
        )
        log.info(f"""
        =========================================
        SIGNAL:       {signal.direction}  ({signal.mode})
        Reason:       {signal.reason}
        Trend:        {signal.trend}
        Confluences:  {signal.confluences}/4
        Entry:        {signal.entry}
        Stop Loss:    {signal.stop_loss}
        Take Profit:  {signal.take_profit}
        R:R Ratio:    {signal.rr_ratio}
        Lot Size:     {lots}
        =========================================
        """)
    else:
        log.info("No signal this bar — goblin waits patiently 🐲")

    log.info("=== Goblin Bot Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Goblin VP bot — single signal check")
    parser.add_argument(
        "--source", choices=["offline", "mt5"], default=None,
        help="Data source (default: Config.DATA_SOURCE / $DATA_SOURCE, i.e. 'offline')",
    )
    args = parser.parse_args()
    run(source=args.source)
