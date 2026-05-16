# src/main.py
import platform
import os
from datetime import datetime, timezone
from src.config import Config
from src.indicators.volume_profile import build_volume_profile
from src.strategy.vp_strategy import generate_signal, calculate_position_size
from src.utils.logger import get_logger

log = get_logger(__name__)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("logs",           exist_ok=True)


def get_provider():
    system = platform.system()

    if system == "Darwin":
        from src.data.csv_provider import CSVProvider
        log.info("Mac — using CSV/DuckDB provider")
        return CSVProvider()

    elif system == "Windows":
        from src.data.mt5_provider import MT5DataProvider
        log.info("Windows — using MT5 provider")
        return MT5DataProvider()

    else:
        raise RuntimeError(f"Unsupported OS: {system}")


def run():
    log.info("=== Goblin Bot Starting 🐲 ===")
    provider = get_provider()

    # --- Fetch Data ---
    # no date range = loads everything available in the DB automatically
    df = provider.get_ohlcv(
        symbol    = Config.SYMBOL,
        timeframe = Config.TIMEFRAME_PROFILE,
    )

    # --- Build Volume Profile ---
    levels = build_volume_profile(df)
    log.info(
        f"POC: {levels.poc:.5f}  |  "
        f"HVNs: {len(levels.hvns)}  |  "
        f"LVNs: {len(levels.lvns)}"
    )

    # --- Visualize ---
    from src.visualizer import plot_volume_profile
    plot_volume_profile(df, levels, symbol=Config.SYMBOL)

    # --- Generate Signal ---
    signal = generate_signal(df, levels)

    if signal:
        lots = calculate_position_size(
            account_balance = Config.ACCOUNT_BALANCE,
            risk_percent    = Config.RISK_PERCENT,
            stop_loss_pips  = abs(signal.entry - signal.stop_loss) / 0.0001
        )
        log.info(f"""
        =====================================
        SIGNAL:      {signal.direction}
        Reason:      {signal.reason}
        Entry:       {signal.entry}
        Stop Loss:   {signal.stop_loss}
        Take Profit: {signal.take_profit}
        R:R Ratio:   {signal.rr_ratio}
        Lot Size:    {lots}
        =====================================
        """)
    else:
        log.info("No signal this bar — goblin waits patiently 🐲")

    log.info("=== Goblin Bot Done ===")


if __name__ == "__main__":
    run()
