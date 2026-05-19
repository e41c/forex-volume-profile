# src/indicators/trend_filter.py
"""
Priority 5 — Trend Filter: Trade With Momentum

Only take longs in uptrends, only take shorts in downtrends.
Uses EMA crossovers and price structure to determine trend.

Three layers of trend confirmation:
  1. EMA 50/200 crossover    — medium term trend direction
  2. Price vs EMA 200        — is price above or below value?
  3. Higher highs/lows       — is price structure bullish or bearish?

All three pointing the same way = strong trend confirmation.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TrendState:
    direction:    str      # "BULLISH", "BEARISH", or "NEUTRAL"
    strength:     int      # 0-3 how many filters confirm the trend
    ema50:        float    # current EMA 50 value
    ema200:       float    # current EMA 200 value
    price_vs_ema: str      # "ABOVE" or "BELOW" EMA200
    structure:    str      # "HIGHER_HIGHS", "LOWER_LOWS", or "RANGING"
    aligned:      bool     # True if signal direction matches trend


def calculate_emas(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate EMA 50 and EMA 200"""
    ema50  = df['Close'].ewm(span=50,  adjust=False).mean()
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    return ema50, ema200


def detect_price_structure(df: pd.DataFrame,
                            lookback: int = 10) -> str:
    """
    Detect if price is making higher highs/lows (bullish)
    or lower highs/lows (bearish) over recent bars.
    """
    if len(df) < lookback * 2:
        return "RANGING"

    recent = df.tail(lookback)
    prior  = df.tail(lookback * 2).head(lookback)

    recent_high = recent['High'].max()
    recent_low  = recent['Low'].min()
    prior_high  = prior['High'].max()
    prior_low   = prior['Low'].min()

    higher_high = recent_high > prior_high
    higher_low  = recent_low  > prior_low
    lower_high  = recent_high < prior_high
    lower_low   = recent_low  < prior_low

    if higher_high and higher_low:
        return "HIGHER_HIGHS"
    elif lower_high and lower_low:
        return "LOWER_LOWS"
    else:
        return "RANGING"


def get_trend_state(df: pd.DataFrame,
                    signal_direction: str = None) -> TrendState:
    """
    Full trend analysis combining EMA, price position, and structure.

    Args:
        df:               OHLCV dataframe
        signal_direction: "BUY" or "SELL" — checks if signal aligns with trend
    """
    if len(df) < 200:
        log.warning(
            f"Only {len(df)} bars available — need 200 for EMA200. "
            f"Trend filter will be lenient."
        )
        return TrendState(
            direction    = "NEUTRAL",
            strength     = 0,
            ema50        = df['Close'].iloc[-1],
            ema200       = df['Close'].iloc[-1],
            price_vs_ema = "NEUTRAL",
            structure    = "RANGING",
            aligned      = True   # don't block signals when not enough data
        )

    ema50, ema200  = calculate_emas(df)
    current_price  = float(df['Close'].iloc[-1])
    current_ema50  = float(ema50.iloc[-1])
    current_ema200 = float(ema200.iloc[-1])

    # ── Layer 1: EMA crossover ──────────────────────────────────
    ema_bullish = current_ema50 > current_ema200
    ema_bearish = current_ema50 < current_ema200

    # ── Layer 2: Price vs EMA200 ────────────────────────────────
    price_above_ema200 = current_price > current_ema200
    price_vs_ema       = "ABOVE" if price_above_ema200 else "BELOW"

    # ── Layer 3: Price structure ────────────────────────────────
    structure = detect_price_structure(df)

    # ── Combine all three layers ────────────────────────────────
    bullish_score = sum([
        ema_bullish,
        price_above_ema200,
        structure == "HIGHER_HIGHS"
    ])
    bearish_score = sum([
        ema_bearish,
        not price_above_ema200,
        structure == "LOWER_LOWS"
    ])

    if bullish_score >= 2:
        direction = "BULLISH"
        strength  = bullish_score
    elif bearish_score >= 2:
        direction = "BEARISH"
        strength  = bearish_score
    else:
        direction = "NEUTRAL"
        strength  = 0

    # ── Check signal alignment ──────────────────────────────────
    if signal_direction is None:
        aligned = True
    elif direction == "NEUTRAL":
        aligned = True    # neutral trend doesn't block signals
    elif signal_direction == "BUY"  and direction == "BULLISH":
        aligned = True
    elif signal_direction == "SELL" and direction == "BEARISH":
        aligned = True
    else:
        aligned = False   # signal goes against the trend

    log.info(
        f"Trend analysis: {direction} (strength {strength}/3)  |  "
        f"EMA50: {current_ema50:.5f}  EMA200: {current_ema200:.5f}  |  "
        f"Price: {price_vs_ema} EMA200  |  "
        f"Structure: {structure}  |  "
        f"Signal aligned: {aligned}"
    )

    return TrendState(
        direction    = direction,
        strength     = strength,
        ema50        = current_ema50,
        ema200       = current_ema200,
        price_vs_ema = price_vs_ema,
        structure    = structure,
        aligned      = aligned
    )


def is_trend_aligned(df: pd.DataFrame,
                     signal_direction: str) -> bool:
    """
    Simple boolean check for use in signal generation.
    Returns True if it's safe to take the signal.
    """
    state = get_trend_state(df, signal_direction)

    if not state.aligned:
        log.info(
            f"Trend filter blocked {signal_direction} signal — "
            f"trend is {state.direction}"
        )

    return state.aligned
