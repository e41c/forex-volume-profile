# src/strategy/vp_strategy.py
"""
Signal generation — Volume Profile Mean Reversion

Trades reversals at institutional volume levels in ranging markets.
The edge: when EURUSD is ranging (NEUTRAL H1 trend), high-volume nodes act
as price magnets. Institutions re-enter at fair value when price tests these levels.

Entry triggers — price must be within the relevant zone of at least one:
  • Rolling 500-bar H1 POC       — the dominant institutional level (5 pips)
  • Rolling profile secondary HVN — MA crossover finds meaningful clusters (5 pips)
  • Daily session POC             — where institutions traded TODAY (15 pips)
  • Value-area edge (VAH/VAL)     — fade the 70% VA boundary back toward POC (5 pips).
                                    Directional: VAH→SELL, VAL→BUY. Needs one more
                                    confluence, so VA fades only fire at a real level.

Weekly and monthly session POCs are context/confluence only — NOT entry triggers.
Backtested: weekly-only entries = 25% win, monthly-only = 28% win → losing trades.
Daily-only entries = 57% win. Daily POC = fresh signal; weekly/monthly = background.

Confluence paths to MIN_CONFLUENCE (3):
  A) rolling_poc + daily_poc + volume
  B) rolling_poc + rolling_hvn + volume   (dual rolling-profile agreement)
  C) daily_poc + session_cluster + volume (daily POC + any other session agrees)
  D) rolling_hvn + daily_poc + volume

Backtested data (23 years, 2003-2026):
  NEUTRAL regime:   60-63% win rate — the edge
  Trending markets: 34% win rate   — VP levels get blown through

Trend-following VP entries were tested (2026-05-30, 2026-06-03) and rejected both times.
  NEUTRAL:          54-60% WR, PF 1.25-1.85 — the edge
  TREND_PULLBACK:   41%   WR, PF 0.82       — losing, VP levels get blown through
NEUTRAL-only filter is load-bearing — do not relax it.

Future: delta proxy (calculate_delta_proxy in trend_filter.py) available for
  soft confluence use; hard-filter version reduced trades 31→11.
"""
import pandas as pd
from dataclasses import dataclass
from src.indicators.volume_profile import VolumeProfileLevels, price_near_level, price_in_cluster
from src.indicators.session_profile import MultiSessionLevels
from src.indicators.trend_filter import (
    get_trend_state, calculate_adx, calculate_atr
)
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TradeSignal:
    direction:    str    # "BUY" or "SELL"
    entry:        float
    stop_loss:    float
    take_profit:  float
    rr_ratio:     float
    reason:       str    # which level triggered, ADX, session score
    trend:        str    # "NEUTRAL" always for this strategy
    mode:         str    # "REVERSION" always for this strategy
    confluences:  int    # number of independent filters confirmed


def calculate_position_size(account_balance: float,
                             risk_percent: float,
                             stop_loss_pips: float,
                             pip_value: float = 10.0) -> float:
    """
    Calculate lot size based on fixed risk %.
    Default pip_value assumes standard lot EURUSD (~$10/pip).
    """
    risk_amount = account_balance * (risk_percent / 100)
    lots        = risk_amount / (stop_loss_pips * pip_value)
    lots        = round(min(lots, 1.0), 2)
    return lots


def volume_confirms_rejection(df: pd.DataFrame,
                               lookback: int = None) -> bool:
    """
    Last bar must have above-average volume.
    High volume at a VP level = institutions are defending/testing it.
    Low volume = weak move, likely to fail.
    """
    lookback = lookback if lookback is not None else Config.VOLUME_LOOKBACK
    if len(df) < lookback:
        return True

    avg_volume  = df['Volume'].tail(lookback).mean()
    last_volume = df['Volume'].iloc[-1]
    confirmed   = last_volume > avg_volume * Config.VOLUME_SPIKE_MULT

    if not confirmed:
        log.debug(
            f"Volume failed — last: {last_volume:.0f}  "
            f"avg: {avg_volume:.0f}  need: {avg_volume * 1.2:.0f}"
        )
    return confirmed


def _session_pocs_near_price(price: float,
                              multi_levels: MultiSessionLevels,
                              pip_size: float) -> list[str]:
    """
    Return list of session names whose POC is within SESSION_POC_ZONE_PIPS of price.
    Uses a wider zone than the rolling profile (15 vs 5 pips) because session POCs
    are built on fewer bars and land in a wider price zone — the old code used 20 pips.
    Includes long_term (2000-bar POC) alongside daily/weekly/monthly.
    """
    nearby = []
    for name, poc in [("long_term", multi_levels.long_term.poc),
                       ("daily",     multi_levels.daily.poc),
                       ("weekly",    multi_levels.weekly.poc),
                       ("monthly",   multi_levels.monthly.poc)]:
        distance = abs(price - poc) / pip_size
        if distance <= Config.SESSION_POC_ZONE_PIPS:
            nearby.append(name)
    return nearby


def generate_signal(df: pd.DataFrame,
                    levels: VolumeProfileLevels,
                    multi_levels: MultiSessionLevels = None,
                    m15_df: pd.DataFrame = None,
                    pip_size: float = 0.0001,
                    entry_bar: pd.Series = None,
                    min_wick_ratio: float = 1.5,
                    _adx: float = None,
                    _trend_direction: str = None,
                    _atr_pips: float = None) -> TradeSignal | None:
    """
    Generate a mean-reversion signal at a VP level in a ranging market.

    entry_bar:        optional M30 bar for candle pattern and SL placement.
                      df (H1) is always used for regime checks (ADX, trend, ATR).
    m15_df:           M30 volume window ending at entry_bar (used for volume check).
    min_wick_ratio:   wick/body ratio threshold. H1 = 1.5, M30 = 1.8 (noisier).
    _adx, _trend_direction, _atr_pips: pre-computed regime values from backtester.
                      Avoids redundant EWM recalculation per M30 sub-bar.
    """
    last  = entry_bar if entry_bar is not None else df.iloc[-1]
    price = float(last['Close'])
    body  = abs(float(last['Close']) - float(last['Open']))
    upper_wick = float(last['High']) - max(float(last['Close']), float(last['Open']))
    lower_wick = min(float(last['Close']), float(last['Open'])) - float(last['Low'])

    # ── Filter 1: Price near a VP institutional level ─────────────
    # Check rolling profile levels first (cheapest — already computed).
    near_rolling_poc = price_near_level(price, levels.poc, pip_size)

    # Non-POC HVN clusters — exclude clusters whose peak is within POC_ZONE_PIPS of POC
    # (POC is already checked above; this avoids double-counting the dominant peak).
    # Entry zone = full cluster mini value area [low, high], NOT just ±5 pips of peak.
    # Wider entry zone = more signals at genuine institutional levels.
    non_poc_clusters = [c for c in levels.hvn_clusters
                        if abs(c.peak - levels.poc) / pip_size > Config.POC_ZONE_PIPS]
    near_rolling_hvn = any(price_in_cluster(price, c) for c in non_poc_clusters)

    # Session POCs — wider zone (15 pips) because session POCs are less precise.
    # Only the DAILY session POC is a valid standalone entry trigger.
    # Weekly/monthly are losing entries alone (25-28% win in backtest).
    # They remain as confluence context — near_session_near counts all sessions.
    session_near = []
    near_daily_poc = False
    if multi_levels is not None:
        session_near = _session_pocs_near_price(price, multi_levels, pip_size)
        near_daily_poc = "daily" in session_near

    session_cluster  = len(session_near) >= 2   # 2+ sessions at same price zone = bonus

    # Value-area edges — fade zones back toward POC (mean reversion to fair value).
    # Only count an edge that sits a meaningful distance from the POC, else it's just
    # the POC zone again. Proximity here is non-directional; the fade DIRECTION is
    # enforced in the confluence step (VAH→SELL, VAL→BUY) once the candle is known.
    near_vah = near_val = False
    if Config.ENABLE_VA_EDGE_FADES:
        vah_dist = (levels.vah - levels.poc) / pip_size
        val_dist = (levels.poc - levels.val) / pip_size
        near_vah = (vah_dist >= Config.VA_EDGE_MIN_DIST_PIPS and
                    abs(price - levels.vah) / pip_size <= Config.POC_ZONE_PIPS)
        near_val = (val_dist >= Config.VA_EDGE_MIN_DIST_PIPS and
                    abs(price - levels.val) / pip_size <= Config.POC_ZONE_PIPS)

    # Entry gate: rolling levels OR today's daily POC OR a value-area edge.
    # Weekly/monthly alone are blocked — not enough signal freshness.
    if not (near_rolling_poc or near_rolling_hvn or near_daily_poc
            or near_vah or near_val):
        return None

    # ── Filter 2: ADX regime — ranging market only ─────────────────
    adx = _adx if _adx is not None else calculate_adx(df)
    if adx > Config.ADX_THRESHOLD:
        log.debug(f"Signal rejected — ADX {adx:.1f} > {Config.ADX_THRESHOLD}")
        return None

    # ── Filter 3: Volume confirmation ─────────────────────────────
    vol_ctx = m15_df if (entry_bar is not None and m15_df is not None
                         and len(m15_df) >= 10) else df
    if not volume_confirms_rejection(vol_ctx):
        log.debug("Signal rejected — volume too low")
        return None

    # ── Filter 4: Rejection candle ────────────────────────────────
    is_bullish_candle = (lower_wick > body * min_wick_ratio and
                         last['Close'] > last['Open'])
    is_bearish_candle = (upper_wick > body * min_wick_ratio and
                         last['Close'] < last['Open'])

    if not (is_bullish_candle or is_bearish_candle):
        return None

    signal_direction = "BUY" if is_bullish_candle else "SELL"

    # ── Filter 5: NEUTRAL regime only ─────────────────────────────
    # Mean reversion works only in ranging markets. Tested (2026-05-30, 2026-06-03):
    # trend pullbacks (BULLISH/BEARISH) = 41% WR, PF 0.82 — VP levels get blown through.
    # NEUTRAL-only = 54-60% WR, PF 1.25-1.85 — load-bearing, do not relax.
    if _trend_direction is not None:
        trend_direction = _trend_direction
    else:
        trend_direction = get_trend_state(df).direction

    if trend_direction != "NEUTRAL":
        log.debug(f"Signal rejected — {trend_direction} regime (NEUTRAL required)")
        return None

    # ── Filter 6: Confluence count ─────────────────────────────────
    # Independent sources of evidence — need ≥ MIN_CONFLUENCE (3).
    # Volume is always True, so need 2 more from the others.
    #
    # Valid paths to 3:
    #   A) rolling_poc + daily_poc + volume
    #   B) rolling_poc + rolling_hvn + volume   (dual rolling-profile agreement)
    #   C) daily_poc + session_cluster + volume (daily + any other session agrees)
    #   D) rolling_hvn + daily_poc + volume
    #   E) va_edge + (any one of the above) + volume   (VA-edge fade at a real level)
    #
    # Value-area edge fade — DIRECTIONAL: VAH rejection is a SELL, VAL rejection is a
    # BUY. Only counts when the rejection candle fades the correct edge toward POC, so
    # a breakout candle (bullish at VAH / bearish at VAL) never earns this confluence.
    va_edge = (near_vah and is_bearish_candle) or (near_val and is_bullish_candle)

    confluences = sum([
        near_rolling_poc,
        near_rolling_hvn,
        near_daily_poc,      # today's institutional level — freshest signal
        session_cluster,     # 2+ sessions (any combo) at same price zone
        va_edge,             # value-area boundary fade in the correct direction
        True,                # volume confirmed above
    ])

    if confluences < Config.MIN_CONFLUENCE:
        log.debug(f"Signal rejected — confluence {confluences} < {Config.MIN_CONFLUENCE}")
        return None

    # ── Filter 7: Minimum SL distance ─────────────────────────────
    # ATR floor: stop must be proportional to current volatility.
    # Pip floor: below MIN_STOP_PIPS, entry costs eat too large a fraction of risk.
    atr_pips    = _atr_pips if _atr_pips is not None else calculate_atr(df) / pip_size
    min_sl_pips = max(atr_pips * Config.MIN_STOP_ATR_MULT, Config.MIN_STOP_PIPS)

    # Build a descriptive label for the triggering level
    if near_rolling_poc:
        level_label = "rolling POC"
    elif near_daily_poc:
        level_label = f"{'/'.join(session_near)} session POC"
    elif near_rolling_hvn:
        level_label = "rolling HVN"
    elif va_edge and near_vah:
        level_label = "VAH fade"
    elif va_edge and near_val:
        level_label = "VAL fade"
    else:
        level_label = "VP level"

    # ── Build trade levels — BUY ───────────────────────────────────
    if is_bullish_candle:
        entry      = price
        stop_loss  = float(last['Low']) - (pip_size * Config.STOP_BUFFER_PIPS)
        sl_pips    = (entry - stop_loss) / pip_size

        if sl_pips < min_sl_pips:
            log.debug(f"Signal rejected — SL {sl_pips:.1f}p < min {min_sl_pips:.1f}p")
            return None

        # TP: next LVN above entry — a structural gap in volume is the natural
        # target price snaps to. Falls back to the MIN_RR formula if none above.
        lvns_above  = [l for l in levels.lvns if l > entry]
        take_profit = (min(lvns_above) if lvns_above
                       else entry + (sl_pips * Config.MIN_RR_RATIO * pip_size))

        tp_pips  = (take_profit - entry) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.debug(f"Signal rejected — R:R {rr_ratio:.2f} < {Config.MIN_RR_RATIO}")
            return None

        if rr_ratio > Config.MAX_RR_RATIO:
            take_profit = entry + (sl_pips * Config.MAX_RR_RATIO * pip_size)
            tp_pips     = (take_profit - entry) / pip_size
            rr_ratio    = round(tp_pips / sl_pips, 2)

        mode_label = "REVERSION" if trend_direction == "NEUTRAL" else "TREND_PULLBACK"
        return TradeSignal(
            direction   = "BUY",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bullish {mode_label.lower()} at {level_label} "
                           f"(ADX {adx:.0f}, trend: {trend_direction}, "
                           f"session: {session_near or 'none'})"),
            trend       = trend_direction,
            mode        = mode_label,
            confluences = confluences,
        )

    # ── Build trade levels — SELL ──────────────────────────────────
    if is_bearish_candle:
        entry      = price
        stop_loss  = float(last['High']) + (pip_size * Config.STOP_BUFFER_PIPS)
        sl_pips    = (stop_loss - entry) / pip_size

        if sl_pips < min_sl_pips:
            log.debug(f"Signal rejected — SL {sl_pips:.1f}p < min {min_sl_pips:.1f}p")
            return None

        # TP: next LVN below entry — symmetric to the BUY case.
        lvns_below  = [l for l in levels.lvns if l < entry]
        take_profit = (max(lvns_below) if lvns_below
                       else entry - (sl_pips * Config.MIN_RR_RATIO * pip_size))

        tp_pips  = (entry - take_profit) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.debug(f"Signal rejected — R:R {rr_ratio:.2f} < {Config.MIN_RR_RATIO}")
            return None

        if rr_ratio > Config.MAX_RR_RATIO:
            take_profit = entry - (sl_pips * Config.MAX_RR_RATIO * pip_size)
            tp_pips     = (entry - take_profit) / pip_size
            rr_ratio    = round(tp_pips / sl_pips, 2)

        mode_label = "REVERSION" if trend_direction == "NEUTRAL" else "TREND_PULLBACK"
        return TradeSignal(
            direction   = "SELL",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bearish {mode_label.lower()} at {level_label} "
                           f"(ADX {adx:.0f}, trend: {trend_direction}, "
                           f"session: {session_near or 'none'})"),
            trend       = trend_direction,
            mode        = mode_label,
            confluences = confluences,
        )

    return None
