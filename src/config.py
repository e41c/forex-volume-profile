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
    # Fixed 200 bins — the proven-best resolution (PF 1.85, 59.5% win, +56.6%).
    # 200 ≥ 55*2 so the MA crossover runs instead of the percentile fallback.
    # On a ~300-pip rolling window that's ~1.5 pips/bin — fine HVN/LVN placement.
    # Fixed (not tick-size) bins are also pair-agnostic: no per-instrument tuning,
    # so the same value carries straight to GBPUSD/EURGBP/JPY profiles.
    # (Tick-size bins were tested 2026-06-03: PF fell to 1.33 and backtests ran
    #  3-4× slower — rejected. See memory project_state.md.)
    PROFILE_BINS          = 200
    POC_ZONE_PIPS         = 5     # proximity threshold — price must be within N pips of level
    SESSION_POC_ZONE_PIPS = 15    # wider zone for session POCs (fewer bars → less precise POC)

    # HVN/LVN detection — dual MA crossover (replaces fixed percentage threshold).
    # Two trailing MAs scan the volume histogram in opposite directions; where they
    # cross marks a structural peak (HVN) or valley (LVN). Ported from MQL4.
    HVN_MA_PERIOD         = 55    # MA period (user-validated value)
    CLUSTER_MERGE_PIPS    = 20    # merge clusters whose peaks are within N pips
    HVN_VALUE_AREA_PCT    = 0.90  # mini value area around each HVN (90% of cluster volume)

    # --- Trade Management ---
    SPREAD_LIMIT_PIPS  = 2.0
    MIN_RR_RATIO       = 1.5
    MAX_RR_RATIO       = 2.0      # mean reversion: quick snap to fair value, not wide targets
    MIN_CONFLUENCE     = 3        # number of independent filters that must agree

    # --- Market Regime (ADX filter) ---
    # Volume profile is a mean-reversion strategy — only works in ranging markets.
    # ADX > threshold = trending = VP levels get blown through = skip.
    # Backtested data: NEUTRAL regime (ADX < 25) = 58-63% win rate.
    #                  Trending markets (ADX > 25) = 34% win rate.
    ADX_THRESHOLD      = 25.0     # skip signals when ADX > 25 (ranging markets only)

    # Trade only London/NY by default. Asian session matters for AUD/NZD/JPY (Tokyo
    # liquidity) — experiment flag, set per-run for those pairs.
    INCLUDE_ASIAN_SESSION = False

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

    # --- Strategy mechanics (centralised, now tunable) ---
    # These were previously hard-coded magic numbers scattered across the strategy
    # and backtester. Values below reproduce the proven-best run EXACTLY — change
    # one, re-run the backtest, and compare. Each is a real knob on the edge.
    VOLUME_LOOKBACK    = 20       # bars averaged for the volume-confirmation check
    # 1.4 chosen by the 2026-06-04 sweep: vs 1.2 → PF 2.05 (was 1.83), DD 7.8% (was
    # 11.3%), Sharpe 5.55 (was 4.74), ~same net on fewer, cleaner trades (33 vs 39).
    VOLUME_SPIKE_MULT  = 1.4      # entry bar volume must exceed avg × this (institutions active)
    STOP_BUFFER_PIPS   = 2.0      # SL placed this many pips beyond the rejection wick
    # 400 chosen by the 2026-06-04 sweep, paired with VOLUME_SPIKE_MULT 1.4: the combo
    # gave PF 2.24, +68.9%, DD 7.8%, Sharpe 6.03 (best on every metric). Window 400
    # surfaces more/fresher levels (more trades, higher net); the volume filter removes
    # the weak ones (window-400 ALONE had DD 14.8% → 7.8% with the filter). They cover
    # each other. NOTE: tuned on EURUSD only — re-validate when multi-pair CSV data lands.
    PROFILE_WINDOW     = 400      # rolling bars used to build each volume profile
    REGIME_WINDOW      = 300      # rolling bars for ADX / trend / ATR regime calc
    SESSION_LONGTERM_BARS = 2000  # window for the long-term session profile (~3 months H1)
    ENTRY_WICK_RATIO   = 1.8      # rejection wick must be ≥ body × this (M30 entries)
    ENTRY_MIN_BODY_PIPS = 1.5     # ignore doji-ish bars below this body size (majors)

    # --- Value-area edge fades ---
    # Fade the value-area boundary back toward POC (mean reversion to fair value).
    # Directional: a rejection at VAH is a SELL, at VAL is a BUY. Same edge as the
    # POC/HVN fades, just at the 70%-value-area boundary — adds entry frequency that
    # carries to every pair. An edge only counts when it sits a meaningful distance
    # from the POC (otherwise it's just the POC zone again, double-counted).
    # REJECTED by the 2026-06-04 sweep: ON → PF 1.83→1.61 with zero clean fade trades
    # (a VA edge alone is <MIN_CONFLUENCE, so it only admits marginal, low-quality
    # trades). Stricter distance (20/30 pips) didn't help. Kept behind the flag, OFF.
    ENABLE_VA_EDGE_FADES  = False
    VA_EDGE_MIN_DIST_PIPS = 10    # VAH/VAL must be ≥ this far from POC to be a distinct zone

    # --- Data ---
    DATA_RAW           = "data/raw"
    DATA_DB            = "data/db/forex.db"

    # Data source — which provider get_provider() returns.
    #   "offline" → CSVProvider (reads DuckDB/parquet) — fast, deterministic backtests.
    #   "mt5"     → MT5DataProvider (live MetaTrader5) — Windows + running terminal.
    # This is an explicit switch, NOT OS auto-detection: the same machine can run
    # offline backtests and live MT5 by flipping DATA_SOURCE, with no code edits.
    # Best practice: pull MT5 history into DuckDB, then backtest "offline" off DuckDB.
    DATA_SOURCE        = os.getenv("DATA_SOURCE", "offline")  # "offline" | "mt5"

    # --- MT5 (Windows only) ---
    MT5_LOGIN          = os.getenv("MT5_LOGIN")
    MT5_PASSWORD       = os.getenv("MT5_PASSWORD")
    MT5_SERVER         = os.getenv("MT5_SERVER")

    # --- Logging ---
    LOG_FILE           = "logs/bot.log"
