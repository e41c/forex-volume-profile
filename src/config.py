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
    HVN_THRESHOLD      = 0.7      # bar volume ≥ 70% of max = high volume node
    LVN_THRESHOLD      = 0.3      # bar volume ≤ 30% of max = low volume node

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

    # --- Minimum stop distance (ATR-based) ---
    # Stops smaller than MIN_STOP_ATR_MULT × ATR(14) get hit by noise.
    MIN_STOP_ATR_MULT  = 0.4      # sl_pips must be ≥ 40% of ATR(14)

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
