# src/strategy/vp_strategy.py
import pandas as pd
from dataclasses import dataclass
from src.indicators.volume_profile import VolumeProfileLevels, price_near_level
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

@dataclass
class TradeSignal:
    direction:   str    # "BUY" or "SELL"
    entry:       float
    stop_loss:   float
    take_profit: float
    rr_ratio:    float
    reason:      str

def calculate_position_size(account_balance: float,
                             risk_percent: float,
                             stop_loss_pips: float,
                             pip_value: float = 10.0) -> float:
    """
    Calculate lot size based on fixed risk %.
    Default pip_value assumes standard lot EURUSD (~$10/pip).
    """
    risk_amount   = account_balance * (risk_percent / 100)
    lots          = risk_amount / (stop_loss_pips * pip_value)
    lots          = round(min(lots, 1.0), 2)  # cap at 1 lot max
    return lots

def generate_signal(df: pd.DataFrame,
                    levels: VolumeProfileLevels,
                    pip_size: float = 0.0001) -> TradeSignal | None:
    """
    Signal logic:
    - Price is near POC or HVN
    - Last candle shows rejection (wick > body)
    - R:R is at least MIN_RR_RATIO
    """
    last  = df.iloc[-1]
    price = last['Close']
    body  = abs(last['Close'] - last['Open'])
    upper_wick = last['High'] - max(last['Close'], last['Open'])
    lower_wick = min(last['Close'], last['Open']) - last['Low']

    near_poc = price_near_level(price, levels.poc, pip_size)
    near_hvn = any(price_near_level(price, h, pip_size) for h in levels.hvns)

    if not (near_poc or near_hvn):
        return None

    # --- Bullish rejection (hammer candle near level) ---
    if lower_wick > body * 1.5 and last['Close'] > last['Open']:
        entry      = price
        stop_loss  = last['Low'] - (pip_size * 2)   # 2 pip buffer
        sl_pips    = (entry - stop_loss) / pip_size

        # target next LVN above, or 2:1 default
        lvns_above = [l for l in levels.lvns if l > entry]
        take_profit = min(lvns_above) if lvns_above else entry + (sl_pips * Config.MIN_RR_RATIO * pip_size)

        tp_pips  = (take_profit - entry) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.info(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        return TradeSignal(
            direction   = "BUY",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = f"Bullish rejection at {'POC' if near_poc else 'HVN'}"
        )

    # --- Bearish rejection (shooting star near level) ---
    if upper_wick > body * 1.5 and last['Close'] < last['Open']:
        entry      = price
        stop_loss  = last['High'] + (pip_size * 2)
        sl_pips    = (stop_loss - entry) / pip_size

        lvns_below = [l for l in levels.lvns if l < entry]
        take_profit = max(lvns_below) if lvns_below else entry - (sl_pips * Config.MIN_RR_RATIO * pip_size)

        tp_pips  = (entry - take_profit) / pip_size
        rr_ratio = round(tp_pips / sl_pips, 2)

        if rr_ratio < Config.MIN_RR_RATIO:
            log.info(f"Signal rejected — R:R {rr_ratio} below minimum")
            return None

        return TradeSignal(
            direction   = "SELL",
            entry       = round(entry, 5),
            stop_loss   = round(stop_loss, 5),
            take_profit = round(take_profit, 5),
            rr_ratio    = rr_ratio,
            reason      = f"Bearish rejection at {'POC' if near_poc else 'HVN'}"
        )

    return None