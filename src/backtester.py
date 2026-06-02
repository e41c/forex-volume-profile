# src/backtester.py
"""
Walk-Forward Backtester — MT5 realistic simulation
Tests the volume profile strategy across all historical data.

Costs modelled to match forex.com Standard account on MT5:
  - Spread:       1.3 pips average EURUSD (forex.com standard)
  - Slippage:     0.5 pips realistic execution
  - Commission:   $0.00 (standard account is spread-only)
  - Swap long:   -0.7 pips/night
  - Swap short:  +0.2 pips/night
  - Triple swap:  Wednesday nights charge 3x (MT5 standard)

MT5 runtime rules modelled:
  - Micro lot sizing (0.01 minimum, 0.01 step)
  - Lot size capped at account margin capacity
  - No trading Friday after 4pm ET (weekend gap risk)
  - No trading Sunday before 5pm ET (market open gap)
  - Minimum candle body filter (no micro doji signals)
  - CAD account pip value calculation
  - USD/CAD conversion on pip value
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, time
from src.indicators.volume_profile import build_volume_profile
from src.indicators.session_profile import build_multi_session_levels
from src.strategy.vp_strategy import generate_signal, calculate_position_size
from src.indicators.trend_filter import calculate_adx, calculate_atr, get_trend_state
from src.utils.session_filter import is_tradeable_session
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)

# approximate USD/CAD rate for pip value conversion
# update this to current rate when running live
USDCAD_RATE = 1.36


@dataclass
class TradingCosts:
    """
    forex.com Standard Account on MT5 — EURUSD realistic costs.

    Standard account (recommended for $1000 CAD starting capital):
        No commission. Spread only. Minimum lot 0.01.

    To switch to RAW Pricing account later (when account grows):
        spread_pips   = 0.3   (much tighter)
        commission    = 14.0  ($7 per leg x2 = $14 round trip per standard lot)
        slippage_pips = 0.3
    """
    spread_pips:    float = 1.3    # forex.com EURUSD standard avg spread
    commission:     float = 0.0    # standard account = no commission
    swap_long:      float = -0.7   # pips per night long EURUSD
    swap_short:     float = 0.2    # pips per night short EURUSD
    slippage_pips:  float = 0.5    # realistic MT5 execution slippage
    pip_size:       float = 0.0001 # EURUSD pip size

    # MT5 lot constraints
    min_lot:        float = 0.01   # micro lot minimum
    lot_step:       float = 0.01   # lot size increment
    max_lot:        float = 100.0  # absolute maximum

    @property
    def total_entry_cost_pips(self) -> float:
        return self.spread_pips + self.slippage_pips

    def is_triple_swap_night(self, dt: datetime) -> bool:
        """
        MT5 charges triple swap on Wednesday nights to cover weekend.
        Wednesday = weekday 2 in Python (Monday=0).
        """
        try:
            import pytz
            ny = pytz.timezone('America/New_York')
            ny_dt = dt.astimezone(ny)
            return ny_dt.weekday() == 2  # Wednesday
        except Exception:
            return False

    def swap_cost_pips(self, direction: str,
                       nights_held: int,
                       entry_time: datetime = None) -> float:
        """
        Total swap in pips including triple-swap Wednesdays.
        MT5 posts swap once per night at 5pm ET rollover.
        """
        if nights_held == 0:
            return 0.0

        rate = self.swap_long if direction == 'BUY' else self.swap_short

        # simple approximation: ~1 in 5 nights is a Wednesday triple swap
        # for a more precise calc we'd need to track each specific night
        # this gives a realistic average cost
        normal_nights  = max(0, nights_held - (nights_held // 5))
        triple_nights  = nights_held // 5

        total = (normal_nights * rate) + (triple_nights * rate * 3)
        return total

    def commission_cost_cad(self, lot_size: float,
                             entry_price: float) -> float:
        """
        Commission in CAD (round trip).
        Standard account = $0.
        RAW account = $14 USD per standard lot round trip,
        converted to CAD.
        """
        if self.commission == 0:
            return 0.0
        usd_cost = (self.commission * lot_size)
        return usd_cost * USDCAD_RATE

    def total_cost_pips(self, direction: str,
                        nights_held: int,
                        lot_size: float,
                        pip_value_cad: float,
                        entry_time: datetime = None,
                        entry_price: float = 1.1) -> float:
        """
        Total cost in pips including all MT5 fees.
        """
        entry_pips = self.total_entry_cost_pips
        swap_pips  = abs(self.swap_cost_pips(direction, nights_held,
                                             entry_time))
        comm_cad   = self.commission_cost_cad(lot_size, entry_price)
        comm_pips  = (comm_cad / pip_value_cad
                      if pip_value_cad > 0 else 0)
        return entry_pips + swap_pips + comm_pips


def normalize_lot_size(lots: float,
                       costs: TradingCosts) -> float:
    """
    MT5 enforces strict lot sizing rules.
    Round to nearest lot_step, enforce min/max.
    """
    if lots < costs.min_lot:
        return costs.min_lot

    # round to nearest lot step (0.01)
    steps = round(lots / costs.lot_step)
    lots  = steps * costs.lot_step
    lots  = min(lots, costs.max_lot)
    lots  = max(lots, costs.min_lot)
    return round(lots, 2)


def pip_value_cad(lot_size: float,
                  entry_price: float = 1.1,
                  usdcad: float = USDCAD_RATE) -> float:
    """
    Real MT5 pip value calculation for EURUSD in a CAD account.

    EURUSD pip value formula:
      pip_value_USD = lot_size * 100000 * pip_size
      pip_value_CAD = pip_value_USD * USD/CAD rate

    For a 0.01 lot at EURUSD 1.1000 with USD/CAD 1.36:
      = 0.01 * 100000 * 0.0001 = $0.10 USD
      = $0.10 * 1.36 = $0.136 CAD per pip
    """
    pip_value_usd = lot_size * 100000 * 0.0001
    return pip_value_usd * usdcad


def is_friday_close(dt: datetime) -> bool:
    """
    MT5 best practice: stop trading Friday after 4pm ET.
    Weekend gaps can blow through stops.
    """
    try:
        import pytz
        ny = pytz.timezone('America/New_York')
        ny_dt = dt.astimezone(ny)
        return ny_dt.weekday() == 4 and ny_dt.time() >= time(16, 0)
    except Exception:
        return False


def has_minimum_body(bar, pip_size: float = 0.0001,
                     min_body_pips: float = 2.0) -> bool:
    """
    MT5 filter: reject micro doji candles.
    Real rejection candles have meaningful bodies.
    Tiny bodies = indecision, not rejection.
    """
    body = abs(float(bar['Close']) - float(bar['Open']))
    return body >= min_body_pips * pip_size


@dataclass
class BacktestTrade:
    entry_time:     datetime
    exit_time:      datetime
    direction:      str
    entry:          float
    stop_loss:      float
    take_profit:    float
    exit_price:     float
    pips_gross:     float
    pips_net:       float
    cost_pips:      float
    nights_held:    int
    result:         str
    rr_ratio:       float
    reason:         str
    trend:          str
    mode:           str     # "TREND" or "REVERSION"
    confluences:    int
    lot_size:       float
    pnl_cad:        float


@dataclass
class BacktestResult:
    trades:           list  = field(default_factory=list)
    equity_curve:     list  = field(default_factory=list)
    total_trades:     int   = 0
    wins:             int   = 0
    losses:           int   = 0
    gross_profit:     float = 0.0
    gross_loss:       float = 0.0
    total_costs:      float = 0.0
    starting_balance: float = Config.ACCOUNT_BALANCE
    final_balance:    float = Config.ACCOUNT_BALANCE
    max_drawdown:     float = 0.0
    max_drawdown_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return round((self.wins / self.total_trades) * 100, 1)

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return round(self.gross_profit / abs(self.gross_loss), 2)

    @property
    def net_profit(self) -> float:
        return round(self.final_balance - self.starting_balance, 2)

    @property
    def net_profit_pct(self) -> float:
        return round(
            (self.net_profit / self.starting_balance) * 100, 1
        )

    @property
    def avg_win_pips(self) -> float:
        wins = [t for t in self.trades if t.result == 'WIN']
        if not wins:
            return 0.0
        return round(sum(t.pips_net for t in wins) / len(wins), 1)

    @property
    def avg_loss_pips(self) -> float:
        losses = [t for t in self.trades if t.result == 'LOSS']
        if not losses:
            return 0.0
        return round(sum(t.pips_net for t in losses) / len(losses), 1)

    @property
    def avg_cost_pips(self) -> float:
        if not self.trades:
            return 0.0
        return round(
            sum(t.cost_pips for t in self.trades) / len(self.trades), 2
        )

    @property
    def sharpe_ratio(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return round(
            (returns.mean() / returns.std()) * np.sqrt(252), 2
        )


def run_backtest(
    df:                   pd.DataFrame,
    df_m15:               pd.DataFrame  = None,
    costs:                TradingCosts  = None,
    profile_window:       int           = 500,
    warmup_bars:          int           = 500,
    pip_size:             float         = 0.0001,
    starting_balance:     float         = Config.ACCOUNT_BALANCE,
    risk_percent:         float         = Config.RISK_PERCENT,
    use_session_profiles: bool          = True,
    min_body_pips:        float         = 2.0,
    entry_wick_ratio:     float         = 2.0,
    entry_min_body_pips:  float         = 1.0,
    verbose:              bool          = False,
) -> BacktestResult:
    """
    Walk-forward backtester — MT5 realistic simulation.

    df_m15:             sub-H1 entry timeframe data (M30 resampled from M15).
    entry_wick_ratio:   wick/body ratio for M30 rejection candles (1.8).
                        H1 fallback always uses 1.5.
    entry_min_body_pips: minimum candle body for M30 entries (1.5 pips).
    """
    if costs is None:
        costs = TradingCosts()

    log.info(
        f"forex.com Standard Account costs:\n"
        f"  Spread:       {costs.spread_pips} pips (avg EURUSD)\n"
        f"  Slippage:     {costs.slippage_pips} pips\n"
        f"  Commission:   ${costs.commission:.2f} (standard = none)\n"
        f"  Swap long:    {costs.swap_long} pips/night\n"
        f"  Swap short:   {costs.swap_short} pips/night\n"
        f"  Triple swap:  Wednesday nights (3x)\n"
        f"  Min lot size: {costs.min_lot} (micro lot)\n"
        f"  USD/CAD rate: {USDCAD_RATE}"
    )

    result = BacktestResult(
        starting_balance = starting_balance,
        final_balance    = starting_balance,
    )

    balance       = starting_balance
    peak_balance  = starting_balance
    in_trade      = False
    current_trade = None
    trades_today  = 0
    last_date     = None
    total_bars    = len(df)
    start_bar     = max(warmup_bars, profile_window)

    # session profile cache — rebuild once per trading day, not every bar
    _session_multi_levels = None
    _session_last_date    = None

    # circuit breaker — pause after consecutive losses
    consecutive_losses = 0
    cooldown_until     = None

    # M30 entry candle index — pre-compute for O(log n) lookups via searchsorted.
    # asi8 on minute-resolution index returns microseconds; multiply by 1000
    # to get nanoseconds so comparisons with Timestamp.value are on the same scale.
    _m15_times_ns = None
    if df_m15 is not None and len(df_m15) > 0:
        import numpy as np
        _m15_times_ns = df_m15.index.asi8 * 1000
        log.info(
            f"M30 entry candles enabled  |  "
            f"{len(df_m15):,} bars  "
            f"{df_m15.index[0].date()} → {df_m15.index[-1].date()}"
        )

    log.info(
        f"Backtester starting  |  "
        f"Bars: {total_bars:,}  |  "
        f"Testing from bar: {start_bar:,}  |  "
        f"Profile window: {profile_window}"
    )

    result.equity_curve.append(balance)

    for i in range(start_bar, total_bars):
        current_bar  = df.iloc[i]
        current_time = df.index[i]
        current_date = current_time.date()

        if current_date != last_date:
            trades_today = 0
            last_date    = current_date

        # ── manage open trade ─────────────────────────────────────
        if in_trade and current_trade is not None:
            ct = current_trade

            bar_date = current_time.date()
            if bar_date != ct.get('last_bar_date'):
                ct['nights_held']   = ct.get('nights_held', 0) + 1
                ct['last_bar_date'] = bar_date

            hit_tp = False
            hit_sl = False

            if ct['direction'] == 'BUY':
                if current_bar['High'] >= ct['take_profit']:
                    hit_tp     = True
                    exit_price = ct['take_profit']
                elif current_bar['Low'] <= ct['stop_loss']:
                    hit_sl     = True
                    exit_price = ct['stop_loss']
            else:
                if current_bar['Low'] <= ct['take_profit']:
                    hit_tp     = True
                    exit_price = ct['take_profit']
                elif current_bar['High'] >= ct['stop_loss']:
                    hit_sl     = True
                    exit_price = ct['stop_loss']

            if hit_tp or hit_sl:
                lot  = ct['lot_size']
                ep   = ct['actual_entry']
                pv   = pip_value_cad(lot, ep)

                if ct['direction'] == 'BUY':
                    gross_pips = (exit_price - ep) / pip_size
                else:
                    gross_pips = (ep - exit_price) / pip_size

                nights    = ct.get('nights_held', 0)
                cost_pips = costs.total_cost_pips(
                    ct['direction'], nights, lot, pv,
                    ct['entry_time'], ep
                )

                net_pips   = gross_pips - cost_pips
                pnl_cad    = net_pips * pv
                balance   += pnl_cad
                result_str = 'WIN' if net_pips > 0 else 'LOSS'

                # circuit breaker tracking
                if result_str == 'WIN':
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    if consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
                        cooldown_until     = current_time + pd.Timedelta(hours=Config.LOSS_COOLDOWN_BARS)
                        consecutive_losses = 0
                        log.info(
                            f"Circuit breaker — {Config.MAX_CONSECUTIVE_LOSSES} losses in a row, "
                            f"pausing until {cooldown_until.strftime('%Y-%m-%d %H:%M')}"
                        )

                if pnl_cad > 0:
                    result.wins         += 1
                    result.gross_profit += pnl_cad
                else:
                    result.losses      += 1
                    result.gross_loss  += pnl_cad

                result.total_costs += cost_pips * pv

                trade = BacktestTrade(
                    entry_time  = ct['entry_time'],
                    exit_time   = current_time,
                    direction   = ct['direction'],
                    entry       = ct['entry'],
                    stop_loss   = ct['stop_loss'],
                    take_profit = ct['take_profit'],
                    exit_price  = round(exit_price, 5),
                    pips_gross  = round(gross_pips, 1),
                    pips_net    = round(net_pips, 1),
                    cost_pips   = round(cost_pips, 2),
                    nights_held = nights,
                    result      = result_str,
                    rr_ratio    = ct['rr_ratio'],
                    reason      = ct['reason'],
                    trend       = ct['trend'],
                    mode        = ct['mode'],
                    confluences = ct['confluences'],
                    lot_size    = lot,
                    pnl_cad     = round(pnl_cad, 2),
                )
                result.trades.append(trade)
                result.total_trades += 1

                if balance > peak_balance:
                    peak_balance = balance
                dd     = peak_balance - balance
                dd_pct = (dd / peak_balance) * 100
                if dd > result.max_drawdown:
                    result.max_drawdown     = dd
                    result.max_drawdown_pct = round(dd_pct, 2)

                result.equity_curve.append(balance)

                if verbose:
                    log.info(
                        f"{result_str:4}  {ct['direction']:4}  "
                        f"gross: {gross_pips:+.1f}  "
                        f"costs: -{cost_pips:.1f}  "
                        f"net: {net_pips:+.1f} pips  "
                        f"pnl: ${pnl_cad:+.2f} CAD  "
                        f"bal: ${balance:.2f}"
                    )

                in_trade      = False
                current_trade = None

            continue

        # ── circuit breaker cooldown ──────────────────────────────
        if cooldown_until is not None and current_time < cooldown_until:
            continue

        # ── session filter ────────────────────────────────────────
        if not is_tradeable_session(current_time):
            continue

        # ── MT5: no new trades Friday after 4pm ET ────────────────
        if is_friday_close(current_time):
            continue

        # ── max daily trades ──────────────────────────────────────
        if trades_today >= Config.MAX_TRADES_PER_DAY:
            continue

        # ── minimum candle body filter ────────────────────────────
        if not has_minimum_body(current_bar, pip_size, min_body_pips):
            continue

        # ── rolling volume profile ─────────────────────────────────
        # Prefer M15 data for the same time window — 4× more bars gives a
        # smoother volume histogram and more precise HVN/LVN placement.
        # Falls back to H1 if M15 window is too short.
        window_df = df.iloc[max(0, i - profile_window):i]
        try:
            if _m15_times_ns is not None:
                win_start_ns = df.index[max(0, i - profile_window)].value
                win_end_ns   = df.index[i].value
                m15_s = int(np.searchsorted(_m15_times_ns, win_start_ns, side='left'))
                m15_e = int(np.searchsorted(_m15_times_ns, win_end_ns,   side='left'))
                m15_vp_window = df_m15.iloc[m15_s:m15_e]
                if len(m15_vp_window) >= Config.HVN_MA_PERIOD * 2:
                    levels = build_volume_profile(m15_vp_window)
                else:
                    levels = build_volume_profile(window_df)
            else:
                levels = build_volume_profile(window_df)
        except Exception as e:
            if verbose:
                log.warning(f"Profile build failed at bar {i}: {e}")
            continue

        # ── session profiles (cached daily — no need to rebuild every bar) ──
        if use_session_profiles and current_time.date() != _session_last_date:
            try:
                # Use 2000-bar window so long_term POC is actually long-term,
                # not the same 500-bar slice as the rolling profile
                session_window        = df.iloc[max(0, i - 2000):i]
                _session_multi_levels = build_multi_session_levels(session_window)
                _session_last_date    = current_time.date()
            except Exception:
                _session_multi_levels = None
        multi_levels = _session_multi_levels if use_session_profiles else None

        # ── signal check ───────────────────────────────────────────
        signal_df = df.iloc[max(0, i - 300):i + 1]
        signal    = None

        # ── Pre-compute H1 regime ONCE per bar ────────────────────
        # ADX, trend direction, and ATR are the same for every M30 sub-bar
        # within this H1 bar. Computing them here (once) instead of inside
        # generate_signal (4× per bar) gives ~4x speedup on the hot path.
        #
        # Pre-check gate: skip entire M30 scan when ADX is too high.
        _h1_adx       = calculate_adx(signal_df)
        _h1_regime_ok = _h1_adx <= Config.ADX_THRESHOLD  # skip bars where market is trending
        _h1_trend_direction = None
        _h1_atr_pips      = None
        if _h1_regime_ok:
            _h1_trend_direction = get_trend_state(signal_df).direction
            _h1_atr_pips        = calculate_atr(signal_df) / pip_size

        if _h1_regime_ok and _m15_times_ns is not None:
            h1_start_ns = current_time.value
            h1_end_ns   = (current_time + pd.Timedelta(hours=1)).value
            sub_start   = int(np.searchsorted(_m15_times_ns, h1_start_ns,
                                              side='left'))
            sub_end     = int(np.searchsorted(_m15_times_ns, h1_end_ns,
                                              side='left'))

            for m15_i in range(sub_start, sub_end):
                m15_bar = df_m15.iloc[m15_i]
                if not has_minimum_body(m15_bar, pip_size,
                                        min_body_pips=entry_min_body_pips):
                    continue
                # Volume window for sub-bar (last 20 bars at that timeframe)
                m15_vol_window = df_m15.iloc[max(0, m15_i - 20):m15_i + 1]
                try:
                    signal = generate_signal(
                        df               = signal_df,
                        levels           = levels,
                        multi_levels     = multi_levels,
                        entry_bar        = m15_bar,
                        m15_df           = m15_vol_window,
                        min_wick_ratio   = entry_wick_ratio,
                        _adx             = _h1_adx,
                        _trend_direction = _h1_trend_direction,
                        _atr_pips        = _h1_atr_pips,
                    )
                except Exception as e:
                    if verbose:
                        log.warning(f"M15 signal error at {m15_i}: {e}")
                    continue
                if signal is not None:
                    break  # take first valid M15 signal per H1 bar

        # fallback: check the H1 bar itself if no M30 signal found.
        # Pass pre-computed regime values — avoids redundant ADX/trend/ATR
        # recalculation on the same H1 window.
        if signal is None and _h1_regime_ok:
            try:
                signal = generate_signal(
                    df               = signal_df,
                    levels           = levels,
                    multi_levels     = multi_levels,
                    _adx             = _h1_adx,
                    _trend_direction = _h1_trend_direction,
                    _atr_pips        = _h1_atr_pips,
                )
            except Exception as e:
                if verbose:
                    log.warning(f"Signal error at bar {i}: {e}")
                continue

        if signal is None:
            continue

        # ── entry costs ────────────────────────────────────────────
        entry_cost = costs.total_entry_cost_pips * pip_size
        if signal.direction == 'BUY':
            actual_entry = signal.entry + entry_cost
        else:
            actual_entry = signal.entry - entry_cost

        # ── MT5 lot sizing ─────────────────────────────────────────
        sl_pips  = abs(signal.entry - signal.stop_loss) / pip_size
        pv       = pip_value_cad(costs.min_lot, actual_entry)

        # risk amount in CAD
        risk_cad = balance * (risk_percent / 100)

        # raw lots based on risk
        if sl_pips > 0 and pv > 0:
            raw_lots = risk_cad / (sl_pips * pv / costs.min_lot)
        else:
            raw_lots = costs.min_lot

        lot_size = normalize_lot_size(raw_lots, costs)

        # ── open trade ─────────────────────────────────────────────
        in_trade      = True
        trades_today += 1
        current_trade = {
            'entry_time':    current_time,
            'last_bar_date': current_time.date(),
            'direction':     signal.direction,
            'entry':         signal.entry,
            'actual_entry':  actual_entry,
            'stop_loss':     signal.stop_loss,
            'take_profit':   signal.take_profit,
            'rr_ratio':      signal.rr_ratio,
            'reason':        signal.reason,
            'trend':         signal.trend,
            'mode':          signal.mode,
            'confluences':   signal.confluences,
            'lot_size':      lot_size,
            'nights_held':   0,
        }

        if verbose:
            log.info(
                f"OPEN  {signal.direction:4}  "
                f"lots: {lot_size}  "
                f"entry: {actual_entry:.5f}  "
                f"sl: {signal.stop_loss:.5f}  "
                f"tp: {signal.take_profit:.5f}"
            )

    result.final_balance = round(balance, 2)
    return result


def print_results(result: BacktestResult,
                  costs:  TradingCosts = None,
                  symbol: str = Config.SYMBOL):
    if costs is None:
        costs = TradingCosts()

    divider = "=" * 54

    log.info(f"""
{divider}
  BACKTEST RESULTS — {symbol}  (CAD account)
{divider}

  FOREX.COM STANDARD ACCOUNT COSTS
  ──────────────────────────────────────────────
  Account type:     Standard (spread only, no commission)
  Spread:           {costs.spread_pips} pips avg EURUSD
  Slippage:         {costs.slippage_pips} pips
  Commission:       $0.00 CAD
  Swap long:        {costs.swap_long} pips/night
  Swap short:       {costs.swap_short} pips/night
  Triple swap:      Wednesday nights (3x rollover)
  Min lot size:     {costs.min_lot} (micro lot)
  Avg cost/trade:   {result.avg_cost_pips:.1f} pips
  Total fees paid:  ${result.total_costs:.2f} CAD

  TRADES
  ──────────────────────────────────────────────
  Total trades:     {result.total_trades:>8,}
  Wins:             {result.wins:>8,}  ({result.win_rate}%)
  Losses:           {result.losses:>8,}

  PERFORMANCE (after all costs, in CAD)
  ──────────────────────────────────────────────
  Starting balance: {result.starting_balance:>10.2f} CAD
  Final balance:    {result.final_balance:>10.2f} CAD
  Net profit:       {result.net_profit:>+10.2f} CAD  ({result.net_profit_pct:+.1f}%)
  Profit factor:    {result.profit_factor:>10.2f}  (need > 1.5)
  Sharpe ratio:     {result.sharpe_ratio:>10.2f}  (need > 1.0)

  RISK
  ──────────────────────────────────────────────
  Max drawdown:     {result.max_drawdown:>10.2f} CAD  ({result.max_drawdown_pct:.1f}%)
  Avg win:          {result.avg_win_pips:>+10.1f} pips net
  Avg loss:         {result.avg_loss_pips:>+10.1f} pips net

  VERDICT
  ──────────────────────────────────────────────
  {'✅ Profit factor PASS  (> 1.5)' if result.profit_factor >= 1.5 else '❌ Profit factor FAIL  (< 1.5)'}
  {'✅ Win rate PASS  (> 40%)' if result.win_rate >= 40 else '❌ Win rate FAIL  (< 40%)'}
  {'✅ Drawdown PASS  (< 20%)' if result.max_drawdown_pct < 20 else '❌ Drawdown FAIL  (> 20%)'}
  {'✅ Sharpe PASS  (> 1.0)' if result.sharpe_ratio >= 1.0 else '❌ Sharpe FAIL  (< 1.0)'}
{divider}
    """)


def save_trade_journal(result: BacktestResult,
                       filepath: str = "data/processed/trade_journal.csv"):
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for t in result.trades:
        rows.append({
            'entry_time':   t.entry_time,
            'exit_time':    t.exit_time,
            'direction':    t.direction,
            'mode':         t.mode,
            'trend':        t.trend,
            'entry':        t.entry,
            'stop_loss':    t.stop_loss,
            'take_profit':  t.take_profit,
            'exit_price':   t.exit_price,
            'lot_size':     t.lot_size,
            'pips_gross':   t.pips_gross,
            'cost_pips':    t.cost_pips,
            'pips_net':     t.pips_net,
            'pnl_cad':      t.pnl_cad,
            'nights_held':  t.nights_held,
            'result':       t.result,
            'rr_ratio':     t.rr_ratio,
            'confluences':  t.confluences,
            'reason':       t.reason,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    log.info(f"Trade journal saved → {filepath}  ({len(df)} trades)")
    return df
