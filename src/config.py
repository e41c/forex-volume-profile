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
    TIMEFRAME_TREND    = "M15"    # trend detection timeframe
    TIMEFRAME_ENTRY    = "M15"    # entry confirmation on M15
    BARS               = 5000
    M15_TREND_BARS     = 300      # M15 bars for trend window (300×15m = 75h ≈ 3 days)

    # --- Volume Profile ---
    PROFILE_BINS       = 100
    POC_ZONE_PIPS      = 5      # was 3 — 3 pips was too tight for H1 bars
    HVN_THRESHOLD      = 0.7
    LVN_THRESHOLD      = 0.3

    # --- Trade Management ---
    SPREAD_LIMIT_PIPS  = 2.0
    MIN_RR_RATIO       = 2.0
    MAX_RR_RATIO       = 4.0    # cap unrealistic targets
    MIN_CONFLUENCE     = 3      # 3 = near level + session OR cluster + volume
    TRAIL_STOP         = True

    # --- Market Regime (ADX filter) ---
    # Volume profile is mean-reversion — only works in ranging markets.
    # ADX > threshold = trending = skip. Best years (2008, 2020) had ranging
    # volatility. Worst years (2014, 2024) were sustained directional trends.
    ADX_THRESHOLD      = 25.0   # skip signals when ADX > 25

    # --- Minimum stop distance (ATR-based) ---
    # Stops smaller than MIN_STOP_ATR_MULT × ATR(14) get hit by noise.
    # In 2024, 8-12 pip stops in a 50+ pip/day market = random SL hits.
    MIN_STOP_ATR_MULT  = 0.4    # sl_pips must be ≥ 40% of ATR(14)

    # --- Circuit breaker ---
    # After N consecutive losses, pause trading for COOLDOWN_BARS H1 bars.
    # Protects capital when the system enters a bad regime.
    MAX_CONSECUTIVE_LOSSES = 3
    LOSS_COOLDOWN_BARS     = 48  # ~2 trading days

    # --- Data ---
    DATA_RAW           = "data/raw"
    DATA_DB            = "data/db/forex.db"

    # --- MT5 (Windows only) ---
    MT5_LOGIN          = os.getenv("MT5_LOGIN")
    MT5_PASSWORD       = os.getenv("MT5_PASSWORD")
    MT5_SERVER         = os.getenv("MT5_SERVER")

    # --- Logging ---
    LOG_FILE           = "logs/bot.log"