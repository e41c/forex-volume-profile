# src/indicators/trend_filter.py
"""
Trend Filter + Delta Proxy

Determines the current H1 trend direction and whether a signal is safe to take.

Two signal modes:
  1. Trend-following  — BUY in BULLISH, SELL in BEARISH.
                        Price pulls back to a VP level then resumes trend direction.
                        Most signals will be this type (more frequent).

  2. Mean reversion   — Both BUY and SELL in NEUTRAL.
                        Price bounces off a VP level back to the centre of range.
                        Kept because NEUTRAL regime still produces clean signals.

Counter-trend signals (BUY in BEARISH, SELL in BULLISH) are blocked — POC levels
get blown through when price has directional conviction.

Delta Proxy (order flow approximation):
  Real order flow requires tick-level bid/ask data. We proxy it from OHLCV:
    bar_delta = (Close - Open) / (High - Low + ε)  → range [-1, +1]
  A rolling average over N bars estimates net buying or selling pressure.
  This adds a direction-quality filter without needing a paid data feed.

ADX (Average Directional Index):
  Measures trend STRENGTH, not direction.
    ADX < 25  → ranging / NEUTRAL       (mean reversion valid)
    ADX 25-35 → moderate trend         (trend-following pullbacks valid)
    ADX > 35  → strong trend            (skip — VP levels get blown through)
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
    aligned:      bool     # True if signal direction is safe to take


def calculate_emas(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate EMA 50 and EMA 200."""
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
    Full trend analysis combining EMA crossover, price vs EMA200, and price structure.

    Args:
        df:               OHLCV dataframe (H1)
        signal_direction: "BUY" or "SELL" — checks if signal is safe to take:
                          - BUY  in BULLISH → aligned (trend-following)
                          - SELL in BEARISH → aligned (trend-following)
                          - BUY  in NEUTRAL → aligned (mean reversion)
                          - SELL in NEUTRAL → aligned (mean reversion)
                          - BUY  in BEARISH → NOT aligned (counter-trend, blocked)
                          - SELL in BULLISH → NOT aligned (counter-trend, blocked)
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
    # Trend-following: signal must match trend direction.
    # Mean reversion: allowed in NEUTRAL (ranging) market.
    # Counter-trend: blocked — POC levels get blown through by directional conviction.
    if signal_direction is None:
        aligned = True
    elif direction == "NEUTRAL":
        aligned = True    # ranging — both mean-reversion directions are fine
    elif direction == "BULLISH" and signal_direction == "BUY":
        aligned = True    # trend-following long
    elif direction == "BEARISH" and signal_direction == "SELL":
        aligned = True    # trend-following short
    else:
        aligned = False   # counter-trend — blocked

    log.debug(
        f"Trend: {direction} ({strength}/3)  |  "
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
    Boolean check for use in signal generation.
    Returns True if the signal direction is safe to take given current trend.
    """
    state = get_trend_state(df, signal_direction)

    if not state.aligned:
        log.debug(
            f"Trend filter blocked {signal_direction} — trend is {state.direction}"
        )

    return state.aligned


def calculate_delta_proxy(df: pd.DataFrame,
                           lookback: int = 5) -> float:
    """
    OHLCV-based buying/selling pressure approximation.

    True order flow = bid/ask volume split per tick (requires tick data).
    This proxy uses bar close position within range as a directional pressure signal:

        bar_delta = (Close - Open) / (High - Low + ε)

    Values: +1.0 = strong buying (closed at top of range)
            -1.0 = strong selling (closed at bottom of range)
             0.0 = indecision / doji

    Returns the rolling mean over `lookback` bars.

    Use:
      delta > +DELTA_PROXY_MIN → confirms BUY pressure
      delta < -DELTA_PROXY_MIN → confirms SELL pressure
    """
    if len(df) < lookback:
        return 0.0

    recent    = df.tail(lookback)
    bar_range = recent['High'] - recent['Low']
    delta     = (recent['Close'] - recent['Open']) / (bar_range + 1e-9)

    return round(float(delta.mean()), 3)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX).

    Measures trend STRENGTH, not direction.
      ADX < 25  — ranging / NEUTRAL        (mean reversion valid)
      ADX 25-35 — moderate trend           (trend-following pullbacks valid)
      ADX > 35  — strong trend             (VP levels get blown through — skip)
      ADX > 50  — very strong trend        (avoid at all costs)

    Returns current ADX value (0-100).
    """
    if len(df) < period * 3:
        return 20.0  # not enough data — assume neutral/ranging

    high  = df['High']
    low   = df['Low']
    close = df['Close']

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    up   = high.diff()
    down = -low.diff()
    dm_plus  = up.where((up > down) & (up > 0),     0.0)
    dm_minus = down.where((down > up) & (down > 0), 0.0)

    alpha    = 1.0 / period
    atr      = tr.ewm(alpha=alpha, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean()  / atr
    di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr

    dx  = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9))
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return round(float(adx.iloc[-1]), 1)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average True Range in price units.
    Used to set a minimum stop-loss distance — stops smaller than
    0.4×ATR get hit by random noise.
    """
    if len(df) < period + 1:
        return float(df['High'].iloc[-1] - df['Low'].iloc[-1])

    high       = df['High']
    low        = df['Low']
    prev_close = df['Close'].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    return round(float(tr.tail(period).mean()), 5)
