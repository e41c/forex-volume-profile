# src/strategy/vp_strategy.py
"""
Signal generation — Volume Profile Mean Reversion

This strategy trades reversals at institutional volume levels (POC, HVN)
in ranging markets. The key insight: when EURUSD is ranging (NEUTRAL H1 trend),
the high-volume nodes act as price magnets. Breaks below them get bought,
breaks above them get sold. This is institutions re-entering at fair value.

Backtested data (23 years, 2003-2026):
  NEUTRAL regime:   63% win rate, +15 pips avg — the edge
  Trending markets: 34% win rate, -4.5 pips avg — counter-productive

Trend-following VP entries were tested (2026-05-30) and rejected:
  ADX 25-35 + BULLISH/BEARISH regime = VP levels don't hold as pullback support.
  Price has directional conviction and blows through the levels.
  The NEUTRAL-only filter is load-bearing — do not relax it.

Filters applied in order (cheapest first):
  1. Price near POC or non-POC HVN                 — at an institutional level
  2. ADX < 25                                       — ranging market only
  3. Volume above 20-bar average                    — institutions defending level
  4. Rejection candle (wick > body × min_wick_ratio) — price refused the level
  5. Session POC confluence                         — multi-TF agreement
  6. NEUTRAL trend filter                           — no directional conviction
  7. Minimum confluence count (≥ 3)                 — enough independent evidence
  8. ATR-based minimum SL distance                  — no noise-level stops
  9. R:R within [MIN_RR_RATIO, MAX_RR_RATIO]        — worthwhile trade

Future: delta proxy (calculate_delta_proxy in trend_filter.py) can be added as
  a soft confluence factor once calibrated. Hard filter version was tested and
  reduced REVERSION trades 31→11 without improving quality.
"""
import pandas as pd
from dataclasses import dataclass
from src.indicators.volume_profile import VolumeProfileLevels, price_near_level
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
    reason:       str
    trend:        str    # trend direction at signal time ("NEUTRAL")
    mode:         str    # "REVERSION" (always, for this strategy)
    confluences:  int    # number of independent filters that confirmed the trade


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
    Volume confirmation — last bar must have above-average volume.
    High volume at a VP level = institutions are defending/testing it.
    Low volume = weak move, likely to fail.
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
    return confirmed


def check_session_confluence(price: float,
                              multi_levels: MultiSessionLevels,
                              pip_size: float = 0.0001,
                              pip_threshold: float = 20.0) -> int:
    """
    Count how many session-timeframe POCs are within pip_threshold of price.
    More timeframes agreeing on the same price zone = stronger institutional level.
    Score ≥ 2 counts as one confluence factor (single TF alignment is noise).
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

    return len(nearby)


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
    _adx:             pre-computed ADX (avoids redundant EWM per M30 sub-bar).
    _trend_direction: pre-computed H1 trend direction ("BULLISH"/"BEARISH"/"NEUTRAL").
                      When provided, skips the internal get_trend_state() call.
    _atr_pips:        pre-computed ATR in pips.
    """
    # Candle-level checks use entry_bar when provided (M30 mode),
    # otherwise fall back to last H1 bar.
    last  = entry_bar if entry_bar is not None else df.iloc[-1]
    price = float(last['Close'])
    body  = abs(float(last['Close']) - float(last['Open']))
    upper_wick = float(last['High']) - max(float(last['Close']), float(last['Open']))
    lower_wick = min(float(last['Close']), float(last['Open'])) - float(last['Low'])

    # ── Filter 1: Price near VP level ─────────────────────────────
    near_poc = price_near_level(price, levels.poc, pip_size)

    # Non-POC HVNs only — the POC is always an HVN, exclude to avoid double-counting
    non_poc_hvns   = [h for h in levels.hvns
                      if abs(h - levels.poc) / pip_size > Config.POC_ZONE_PIPS]
    near_extra_hvn = any(price_near_level(price, h, pip_size)
                         for h in non_poc_hvns)

    if not (near_poc or near_extra_hvn):
        return None

    # ── Filter 2: ADX regime — ranging market only ─────────────────
    # ADX > 25 means trending — VP levels get blown through.
    # Backtested: NEUTRAL (ADX < 25) = 58-63% win. Trending = 34% win.
    adx = _adx if _adx is not None else calculate_adx(df)
    if adx > Config.ADX_THRESHOLD:
        log.debug(f"Signal rejected — ADX {adx:.1f} > {Config.ADX_THRESHOLD} (trending)")
        return None

    # ── Filter 3: Volume confirmation ─────────────────────────────
    # Use M30 volume window when checking an M30 entry bar.
    vol_ctx = m15_df if (entry_bar is not None and m15_df is not None
                         and len(m15_df) >= 10) else df
    if not volume_confirms_rejection(vol_ctx):
        log.debug("Signal rejected — volume too low at key level")
        return None

    # ── Filter 4: Rejection candle ────────────────────────────────
    # Bullish: large lower wick (buyers defending level) with bullish close
    # Bearish: large upper wick (sellers defending level) with bearish close
    is_bullish_candle = (lower_wick > body * min_wick_ratio and
                         last['Close'] > last['Open'])
    is_bearish_candle = (upper_wick > body * min_wick_ratio and
                         last['Close'] < last['Open'])

    if not (is_bullish_candle or is_bearish_candle):
        return None

    signal_direction = "BUY" if is_bullish_candle else "SELL"

    # ── Filter 5: Session POC confluence ──────────────────────────
    session_score = 0
    if multi_levels is not None:
        session_score = check_session_confluence(price, multi_levels, pip_size)

    # ── Filter 6: NEUTRAL trend filter ────────────────────────────
    # Only trade when H1 trend is NEUTRAL (ranging).
    # BULLISH and BEARISH regimes = VP levels don't hold. Skip.
    if _trend_direction is not None:
        trend_direction = _trend_direction
    else:
        trend_direction = get_trend_state(df).direction

    if trend_direction != "NEUTRAL":
        log.debug(f"Signal rejected — trend is {trend_direction} (need NEUTRAL)")
        return None

    # ── Filter 7: Confluence count ─────────────────────────────────
    # Four independent sources of evidence — need ≥ MIN_CONFLUENCE (3).
    # Valid paths to 3 confluences:
    #   A) near_poc + session_score≥2 + volume  — POC with strong TF alignment
    #   B) near_extra_hvn + session_score≥2 + volume  — secondary HVN with TF backing
    #   C) near_poc + near_extra_hvn + volume  — dual-level agreement (no session needed)
    #      This path fires more often now because MA crossover finds more valid HVNs.
    #
    # session_score ≥ 2 kept for paths A and B — single TF alignment is noise
    # (data: score=1 avg -2.45 pips). Path C bypasses the session requirement
    # because overlapping POC + HVN is independently strong evidence.
    confluences = sum([
        near_poc,
        near_extra_hvn,
        session_score >= 2,   # ≥2 TF POCs within 20 pips — strong institutional memory
        True,                 # volume confirmed above (always True at this point)
    ])

    if confluences < Config.MIN_CONFLUENCE:
        log.debug(
            f"Signal rejected — confluence {confluences} < {Config.MIN_CONFLUENCE}"
        )
        return None

    # ── Filter 8: Minimum SL distance (two independent floors) ───
    # Floor 1 (ATR): stop must be large enough relative to recent volatility.
    # Floor 2 (pips): absolute minimum — below MIN_STOP_PIPS, entry costs
    #   (1.8 pips spread+slippage) consume an unacceptable fraction of the risk.
    atr_pips    = _atr_pips if _atr_pips is not None else calculate_atr(df) / pip_size
    min_sl_pips = max(atr_pips * Config.MIN_STOP_ATR_MULT, Config.MIN_STOP_PIPS)

    reason_level = "POC" if near_poc else "HVN cluster"

    # ── Build trade levels — BUY ───────────────────────────────────
    if is_bullish_candle:
        entry      = price
        stop_loss  = float(last['Low']) - (pip_size * 2)
        sl_pips    = (entry - stop_loss) / pip_size

        if sl_pips < min_sl_pips:
            log.debug(
                f"Signal rejected — SL {sl_pips:.1f} pips < "
                f"min {min_sl_pips:.1f} (40% of ATR {atr_pips:.1f})"
            )
            return None

        lvns_above  = [l for l in levels.lvns if l > entry]
        take_profit = (min(lvns_above) if lvns_above
                       else entry + (sl_pips * Config.MIN_RR_RATIO * pip_size))

        tp_pips  = (take_profit - entry) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.debug(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        # ── Filter 9: Cap TP at MAX_RR_RATIO ──────────────────────
        if rr_ratio > Config.MAX_RR_RATIO:
            take_profit = entry + (sl_pips * Config.MAX_RR_RATIO * pip_size)
            tp_pips     = (take_profit - entry) / pip_size
            rr_ratio    = round(tp_pips / sl_pips, 2)

        return TradeSignal(
            direction   = "BUY",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bullish reversion at {reason_level} "
                           f"(ADX {adx:.0f}, session {session_score})"),
            trend       = trend_direction,
            mode        = "REVERSION",
            confluences = confluences,
        )

    # ── Build trade levels — SELL ──────────────────────────────────
    if is_bearish_candle:
        entry      = price
        stop_loss  = float(last['High']) + (pip_size * 2)
        sl_pips    = (stop_loss - entry) / pip_size

        if sl_pips < min_sl_pips:
            log.debug(
                f"Signal rejected — SL {sl_pips:.1f} pips < "
                f"min {min_sl_pips:.1f} (40% of ATR {atr_pips:.1f})"
            )
            return None

        lvns_below  = [l for l in levels.lvns if l < entry]
        take_profit = (max(lvns_below) if lvns_below
                       else entry - (sl_pips * Config.MIN_RR_RATIO * pip_size))

        tp_pips  = (entry - take_profit) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.debug(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        if rr_ratio > Config.MAX_RR_RATIO:
            take_profit = entry - (sl_pips * Config.MAX_RR_RATIO * pip_size)
            tp_pips     = (entry - take_profit) / pip_size
            rr_ratio    = round(tp_pips / sl_pips, 2)

        return TradeSignal(
            direction   = "SELL",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bearish reversion at {reason_level} "
                           f"(ADX {adx:.0f}, session {session_score})"),
            trend       = trend_direction,
            mode        = "REVERSION",
            confluences = confluences,
        )

    return None
