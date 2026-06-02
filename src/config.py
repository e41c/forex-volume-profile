# src/config.py
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # --- Account ---
    ACCOUNT_BALANCE    = 5000.00   # CAD — starting capital
    RISK_PERCENT       = 3.0       # % of balance risked per trade
    MAX_TRADES_PER_DAY = 3

    # --- Instrument ---
    SYMBOL             = "EURUSD"
    TIMEFRAME_PROFILE  = "H1"      # build volume profile on H1
    TIMEFRAME_TREND    = "M15"     # raw data timeframe (resampled to M30 for entries)
    BARS               = 5000

    # --- Volume Profile ---
    PROFILE_BINS       = 100
    POC_ZONE_PIPS      = 5        # proximity threshold — price must be within N pips of level

    # HVN/LVN detection — dual MA crossover (replaces fixed percentage threshold)
    # Two trailing MAs scan the volume histogram in opposite directions.
    # Where they cross → structural peak (HVN) or valley (LVN).
    # More sensitive than fixed threshold — finds all meaningful clusters.
    HVN_MA_PERIOD      = 55       # MA period (user-validated value)
    CLUSTER_MERGE_PIPS = 10       # merge two clusters if their peaks are within N pips

    # --- Trade Management ---
    SPREAD_LIMIT_PIPS  = 2.0
    MIN_RR_RATIO       = 1.5
    MAX_RR_RATIO       = 2.0      # frequent small wins > rare large wins
    MIN_CONFLUENCE     = 3        # number of independent filters that must agree

    # --- Market Regime (ADX filter) ---
    # Volume profile is a mean-reversion strategy — only works in ranging markets.
    # ADX > threshold = trending = VP levels get blown through = skip.
    # Backtested data: NEUTRAL regime (ADX < 25) = 58-63% win rate.
    #                  Trending markets (ADX > 25) = 34% win rate.
    ADX_THRESHOLD      = 25.0     # skip signals when ADX > 25

    # --- Stop distance floors ---
    # Two independent floors — both must pass.
    # ATR floor: stop must be large enough relative to recent volatility.
    # Pip floor: stop must never be so small that spread+slippage consumes the trade.
    MIN_STOP_ATR_MULT  = 0.4      # sl_pips ≥ 40% of ATR(14)
    MIN_STOP_PIPS      = 8.0      # absolute floor — below this, 1.8-pip entry cost eats 22%+ of risk

    # --- Circuit breaker ---
    # After N consecutive losses, pause trading for COOLDOWN_BARS H1 bars.
    MAX_CONSECUTIVE_LOSSES = 3
    LOSS_COOLDOWN_BARS     = 48   # ~2 trading days

    # --- Data ---
    DATA_RAW           = "data/raw"
    DATA_DB            = "data/db/forex.db"

    # --- MT5 (Windows only) ---
    MT5_LOGIN          = os.getenv("MT5_LOGIN")
    MT5_PASSWORD       = os.getenv("MT5_PASSWORD")
    MT5_SERVER         = os.getenv("MT5_SERVER")

    # --- Logging ---
    LOG_FILE           = "logs/bot.log"
