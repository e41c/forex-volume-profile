# src/config.py
import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    # --- Account ---
    ACCOUNT_BALANCE    = 1000.00
    RISK_PERCENT       = 1.0
    MAX_TRADES_PER_DAY = 3

    # --- Instrument ---
    SYMBOL             = "EURUSD"
    TIMEFRAME_PROFILE  = "H1"     # build volume profile on H1
    TIMEFRAME_ENTRY    = "M15"    # entry confirmation on M15
    BARS               = 5000

    # --- Volume Profile ---
    PROFILE_BINS       = 100
    POC_ZONE_PIPS      = 15
    HVN_THRESHOLD      = 0.5
    LVN_THRESHOLD      = 0.3

    # --- Trade Management ---
    SPREAD_LIMIT_PIPS  = 2.0
    MIN_RR_RATIO       = 2.0
    TRAIL_STOP         = True

    # --- Data ---
    DATA_RAW           = "data/raw"
    DATA_DB            = "data/db/forex.db"

    # --- MT5 (Windows only) ---
    MT5_LOGIN          = os.getenv("MT5_LOGIN")
    MT5_PASSWORD       = os.getenv("MT5_PASSWORD")
    MT5_SERVER         = os.getenv("MT5_SERVER")

    # --- Logging ---
    LOG_FILE           = "logs/bot.log"