# src/strategy/vp_strategy.py
import pandas as pd
from dataclasses import dataclass
from src.indicators.volume_profile import VolumeProfileLevels, price_near_level
from src.indicators.session_profile import MultiSessionLevels, poc_confluence
from src.indicators.trend_filter import (
    is_trend_aligned, get_trend_state, calculate_adx, calculate_atr
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
    reason:       str
    trend:        str    # trend state at signal time
    confluences:  int    # how many filters confirmed this trade


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
    lots        = round(min(lots, 1.0), 2)  # cap at 1 lot max
    return lots


def volume_confirms_rejection(df: pd.DataFrame,
                               lookback: int = 20) -> bool:
    """
    Priority 3 — Volume Confirmation.
    Rejection candle must have above-average volume.
    High volume = institutions defending the level.
    Low volume  = weak move, likely to fail.
    """
    if len(df) < lookback:
        return True  # not enough data, don't block

    avg_volume  = df['Volume'].tail(lookback).mean()
    last_volume = df['Volume'].iloc[-1]
    confirmed   = last_volume > avg_volume * 1.2

    if not confirmed:
        log.debug(
            f"Volume confirmation failed — "
            f"last: {last_volume:.0f}  avg: {avg_volume:.0f}  "
            f"need: {avg_volume * 1.2:.0f}"
        )
    else:
        log.debug(
            f"Volume confirmed — "
            f"last: {last_volume:.0f}  avg: {avg_volume:.0f}"
        )
    return confirmed


def check_session_confluence(price: float,
                              multi_levels: MultiSessionLevels,
                              pip_size: float = 0.0001,
                              pip_threshold: float = 20.0) -> int:
    """
    Priority 4 — Session Profile Confluence.
    Count how many session POCs are within pip_threshold of price.
    More POCs clustered near price = stronger level.
    """
    pocs = {
        "long_term": multi_levels.long_term.poc,
        "monthly":   multi_levels.monthly.poc,
        "weekly":    multi_levels.weekly.poc,
        "daily":     multi_levels.daily.poc,
    }

    nearby = []
    for name, poc in pocs.items():
        dist_pips = abs(price - poc) / pip_size
        if dist_pips <= pip_threshold:
            nearby.append(name)

    if nearby:
        log.debug(f"Session confluence — POCs nearby: {nearby}")
    else:
        log.debug("No session POC confluence near current price")

    return len(nearby)


def generate_signal(df: pd.DataFrame,
                    levels: VolumeProfileLevels,
                    multi_levels: MultiSessionLevels = None,
                    m15_df: pd.DataFrame = None,
                    pip_size: float = 0.0001) -> TradeSignal | None:
    """
    Full signal logic with all fixes applied:

    1. Price near POC or HVN
    2. ADX regime filter — ranging market only (ADX < 25)
    3. Rejection candle (wick > body)
    4. Volume above average at the level
    5. Session POC confluence check
    6. Trend filter on M15 (or H1 fallback) — NEUTRAL only
    7. Minimum confluence 3
    8. ATR-based minimum SL — no noise-level stops
    9. R:R between 2:1 and 4:1

    m15_df: M15 bars for trend + ADX detection (higher frequency NEUTRAL signals).
            Falls back to df (H1) if not provided.
    """
    last  = df.iloc[-1]
    price = float(last['Close'])
    body  = abs(float(last['Close']) - float(last['Open']))
    upper_wick = float(last['High']) - max(float(last['Close']),
                                           float(last['Open']))
    lower_wick = min(float(last['Close']),
                     float(last['Open'])) - float(last['Low'])

    near_poc = price_near_level(price, levels.poc, pip_size)

    # HVNs that are NOT the POC itself — true cluster confirmation
    # (POC is always an HVN, so near_hvn was double-counting near_poc)
    non_poc_hvns   = [h for h in levels.hvns
                      if abs(h - levels.poc) / pip_size > Config.POC_ZONE_PIPS]
    near_extra_hvn = any(price_near_level(price, h, pip_size)
                         for h in non_poc_hvns)

    if not (near_poc or near_extra_hvn):
        return None

    # ADX market regime filter on H1 — skip when market is trending
    # M15 ADX was tried but is always < 25 (3.5h too short for meaningful trend strength).
    # M15 is reserved for future entry candle work (tighter stops), not regime detection.
    adx = calculate_adx(df)
    if adx > Config.ADX_THRESHOLD:
        log.debug(f"Signal rejected — ADX {adx:.1f} > {Config.ADX_THRESHOLD} (trending)")
        return None

    # volume confirmation
    if not volume_confirms_rejection(df):
        log.debug("Signal rejected — volume too low at key level")
        return None

    # detect candle direction
    is_bullish_candle = (lower_wick > body * 1.5 and
                         last['Close'] > last['Open'])
    is_bearish_candle = (upper_wick > body * 1.5 and
                         last['Close'] < last['Open'])

    if not (is_bullish_candle or is_bearish_candle):
        return None

    signal_direction = "BUY" if is_bullish_candle else "SELL"

    # session confluence
    session_score = 0
    if multi_levels is not None:
        session_score = check_session_confluence(
            price, multi_levels, pip_size
        )

    # trend alignment on H1 — NEUTRAL only (data: H1 NEUTRAL = 57% win, trending = losing)
    if not is_trend_aligned(df, signal_direction):
        log.debug(f"Signal rejected — {signal_direction} goes against trend")
        return None

    # count independent confluences:
    #   near_poc       — price at highest-volume node
    #   near_extra_hvn — price also near a separate HVN cluster (independent of POC)
    #   session_score  — multi-timeframe POC alignment (need ≥ 2 TFs to count)
    #   volume         — always True (hard-checked above, confirmed institutional interest)
    #
    # session_score >= 2 required to count: data shows score=1 averages -2.45 pips,
    # score=2 averages +2.06 pips — single TF alignment is noise, not signal
    confluences = sum([
        near_poc,
        near_extra_hvn,
        session_score >= 2,
        True,   # volume already confirmed above
    ])

    if confluences < Config.MIN_CONFLUENCE:
        log.debug(
            f"Signal rejected — confluence {confluences} "
            f"below minimum {Config.MIN_CONFLUENCE}"
        )
        return None

    # ATR from H1 df — stop sizing relative to the level timeframe's volatility
    atr         = calculate_atr(df)
    atr_pips    = atr / pip_size
    min_sl_pips = atr_pips * Config.MIN_STOP_ATR_MULT

    # ── Bullish rejection ─────────────────────────────────────────
    if is_bullish_candle:
        entry      = price
        stop_loss  = float(last['Low']) - (pip_size * 2)
        sl_pips    = (entry - stop_loss) / pip_size

        # Minimum stop distance — stops smaller than 0.4×ATR get hit by noise
        if sl_pips < min_sl_pips:
            log.debug(
                f"Signal rejected — SL {sl_pips:.1f} pips < "
                f"min {min_sl_pips:.1f} (40% of ATR {atr_pips:.1f})"
            )
            return None

        lvns_above  = [l for l in levels.lvns if l > entry]
        take_profit = (min(lvns_above) if lvns_above
                       else entry + (sl_pips * Config.MIN_RR_RATIO
                                     * pip_size))

        tp_pips  = (take_profit - entry) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.debug(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        # FIX 2 — cap unrealistic targets
        # diagnosis showed rr_ratio > 4 almost always hit SL not TP
        if rr_ratio > Config.MAX_RR_RATIO:
            take_profit = entry + (sl_pips * Config.MAX_RR_RATIO * pip_size)
            tp_pips     = (take_profit - entry) / pip_size
            rr_ratio    = round(tp_pips / sl_pips, 2)
            log.debug(f"TP capped at {Config.MAX_RR_RATIO}:1 → {take_profit:.5f}")

        trend_state = get_trend_state(df, "BUY")

        return TradeSignal(
            direction   = "BUY",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bullish rejection + vol confirm + trend aligned "
                           f"at {'POC' if near_poc else 'HVN cluster'} "
                           f"(session score: {session_score})"),
            trend       = trend_state.direction,
            confluences = confluences,
        )

    # ── Bearish rejection ─────────────────────────────────────────
    if is_bearish_candle:
        entry      = price
        stop_loss  = float(last['High']) + (pip_size * 2)
        sl_pips    = (stop_loss - entry) / pip_size

        # Minimum stop distance — stops smaller than 0.4×ATR get hit by noise
        if sl_pips < min_sl_pips:
            log.debug(
                f"Signal rejected — SL {sl_pips:.1f} pips < "
                f"min {min_sl_pips:.1f} (40% of ATR {atr_pips:.1f})"
            )
            return None

        lvns_below  = [l for l in levels.lvns if l < entry]
        take_profit = (max(lvns_below) if lvns_below
                       else entry - (sl_pips * Config.MIN_RR_RATIO
                                     * pip_size))

        tp_pips  = (entry - take_profit) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.debug(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        # FIX 2 — cap unrealistic targets
        if rr_ratio > Config.MAX_RR_RATIO:
            take_profit = entry - (sl_pips * Config.MAX_RR_RATIO * pip_size)
            tp_pips     = (entry - take_profit) / pip_size
            rr_ratio    = round(tp_pips / sl_pips, 2)
            log.debug(f"TP capped at {Config.MAX_RR_RATIO}:1 → {take_profit:.5f}")

        trend_state = get_trend_state(df, "SELL")

        return TradeSignal(
            direction   = "SELL",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bearish rejection + vol confirm + trend aligned "
                           f"at {'POC' if near_poc else 'HVN cluster'} "
                           f"(session score: {session_score})"),
            trend       = trend_state.direction,
            confluences = confluences,
        )

    return None
