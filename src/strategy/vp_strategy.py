# src/strategy/vp_strategy.py
import pandas as pd
from dataclasses import dataclass
from src.indicators.volume_profile import VolumeProfileLevels, price_near_level
from src.indicators.session_profile import MultiSessionLevels, poc_confluence
from src.indicators.trend_filter import is_trend_aligned, get_trend_state
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
        log.info(
            f"Volume confirmation failed — "
            f"last: {last_volume:.0f}  avg: {avg_volume:.0f}  "
            f"need: {avg_volume * 1.2:.0f}"
        )
    else:
        log.info(
            f"Volume confirmed✅ — "
            f"last: {last_volume:.0f}  avg: {avg_volume:.0f}"
        )
    return confirmed


def check_session_confluence(price: float,
                              multi_levels: MultiSessionLevels,
                              pip_size: float = 0.0001,
                              pip_threshold: float = 20.0) -> int:
    """
    Priority 4 — Session Profile Confluence.
    Count how many session POCs (daily/weekly/monthly/longterm)
    are within pip_threshold of current price.
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
        log.info(f"Session confluence✅ — POCs nearby: {nearby}")
    else:
        log.info("No session POC confluence near current price")

    return len(nearby)


def generate_signal(df: pd.DataFrame,
                    levels: VolumeProfileLevels,
                    multi_levels: MultiSessionLevels = None,
                    pip_size: float = 0.0001) -> TradeSignal | None:
    """
    Full signal logic with all five priority filters:

    1. Price near POC or HVN
    2. Rejection candle (wick > body)
    3. Volume above average at the level       <- Priority 3
    4. Session POC confluence check            <- Priority 4
    5. Trend filter - signal aligns with EMA   <- Priority 5
    6. R:R minimum 2:1
    """
    last  = df.iloc[-1]
    price = float(last['Close'])
    body  = abs(float(last['Close']) - float(last['Open']))
    upper_wick = float(last['High']) - max(float(last['Close']),
                                           float(last['Open']))
    lower_wick = min(float(last['Close']),
                     float(last['Open'])) - float(last['Low'])

    near_poc = price_near_level(price, levels.poc, pip_size)
    near_hvn = any(price_near_level(price, h, pip_size)
                   for h in levels.hvns)

    if not (near_poc or near_hvn):
        return None

    # Priority 3: volume confirmation
    if not volume_confirms_rejection(df):
        log.info("Signal rejected — volume too low at key level")
        return None

    # detect candle direction before trend check
    is_bullish_candle = (lower_wick > body * 1.5 and
                         last['Close'] > last['Open'])
    is_bearish_candle = (upper_wick > body * 1.5 and
                         last['Close'] < last['Open'])

    if not (is_bullish_candle or is_bearish_candle):
        return None  # no rejection candle

    signal_direction = "BUY" if is_bullish_candle else "SELL"

    # Priority 4: session confluence
    session_score = 0
    if multi_levels is not None:
        session_score = check_session_confluence(
            price, multi_levels, pip_size
        )

    # Priority 5: trend alignment
    if not is_trend_aligned(df, signal_direction):
        log.info(
            f"Signal rejected — {signal_direction} goes against trend"
        )
        return None

    # count total confluences for logging
    confluences = sum([
        near_poc,
        near_hvn,
        session_score > 0,
        True,  # volume already confirmed above
    ])

    # Bullish rejection (hammer candle near level)
    if is_bullish_candle:
        entry      = price
        stop_loss  = float(last['Low']) - (pip_size * 2)
        sl_pips    = (entry - stop_loss) / pip_size

        lvns_above  = [l for l in levels.lvns if l > entry]
        take_profit = (min(lvns_above) if lvns_above
                       else entry + (sl_pips * Config.MIN_RR_RATIO
                                     * pip_size))

        tp_pips  = (take_profit - entry) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.info(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        trend_state = get_trend_state(df, "BUY")

        return TradeSignal(
            direction   = "BUY",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bullish rejection + vol confirm + trend aligned "
                           f"at {'POC' if near_poc else 'HVN'} "
                           f"(session score: {session_score})"),
            trend        = trend_state.direction,
            confluences  = confluences,
        )

    # Bearish rejection (shooting star near level)
    if is_bearish_candle:
        entry      = price
        stop_loss  = float(last['High']) + (pip_size * 2)
        sl_pips    = (stop_loss - entry) / pip_size

        lvns_below  = [l for l in levels.lvns if l < entry]
        take_profit = (max(lvns_below) if lvns_below
                       else entry - (sl_pips * Config.MIN_RR_RATIO
                                     * pip_size))

        tp_pips  = (entry - take_profit) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.info(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        trend_state = get_trend_state(df, "SELL")

        return TradeSignal(
            direction   = "SELL",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = (f"Bearish rejection + vol confirm + trend aligned "
                           f"at {'POC' if near_poc else 'HVN'} "
                           f"(session score: {session_score})"),
            trend        = trend_state.direction,
            confluences  = confluences,
        )

    return None
