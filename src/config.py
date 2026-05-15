# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- Account ---
    ACCOUNT_BALANCE    = 1000.00   # CAD starting balance
    RISK_PERCENT       = 1.0       # % risked per trade — DO NOT CHANGE
    MAX_TRADES_PER_DAY = 3         # goblin discipline

    # --- Instrument ---
    SYMBOL             = "EURUSD"
    TIMEFRAME_PROFILE  = "H1"      # build volume profile on H1
    TIMEFRAME_ENTRY    = "M15"     # look for entries on M15
    BARS               = 2000      # how many bars to build profile from

    # --- Volume Profile ---
    PROFILE_BINS       = 100       # price bucket resolution
    POC_ZONE_PIPS      = 3         # how close to POC counts as "at POC"
    HVN_THRESHOLD      = 0.7       # top 70% volume = High Volume Node
    LVN_THRESHOLD      = 0.3       # bottom 30% volume = Low Volume Node

    # --- Trade Management ---
    SPREAD_LIMIT_PIPS  = 2.0       # skip trade if spread too wide
    MIN_RR_RATIO       = 2.0       # minimum reward:risk ratio (2:1)
    TRAIL_STOP         = True      # trail stop to breakeven at 1:1

    # --- MT5 (Windows only) ---
    MT5_LOGIN          = os.getenv("MT5_LOGIN")
    MT5_PASSWORD       = os.getenv("MT5_PASSWORD")
    MT5_SERVER         = os.getenv("MT5_SERVER")

    # --- Paths ---
    DATA_RAW           = "data/raw"
    DATA_PROCESSED     = "data/processed"
    LOG_FILE           = "logs/bot.log"