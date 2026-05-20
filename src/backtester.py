# src/backtester.py
"""
Walk-Forward Backtester — with realistic trading costs
Tests the volume profile strategy across all historical data.
Simulates exactly what the live bot will do — bar by bar.

Costs modelled:
  - Spread:      bid/ask difference paid on entry
  - Commission:  per-lot fee paid on open and close
  - Swap:        overnight interest charge per night held
  - Slippage:    execution price movement on entry
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from src.indicators.volume_profile import build_volume_profile
from src.indicators.session_profile import build_multi_session_levels
from src.strategy.vp_strategy import generate_signal, calculate_position_size
from src.config import Config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TradingCosts:
    """
    Realistic forex.com trading costs for EURUSD.
    Adjust these to match your exact account type.

    Standard account (no commission, wider spread):
        spread_pips   = 1.2
        commission    = 0.0
        swap_long     = -0.8
        swap_short    = 0.2
        slippage_pips = 0.5

    ECN/RAW account (commission, tight spread):
        spread_pips   = 0.2
        commission    = 6.0  (per standard lot round trip)
        swap_long     = -0.8
        swap_short    = 0.2
        slippage_pips = 0.3
    """
    spread_pips:   float = 1.2    # pips lost on entry (bid/ask)
    commission:    float = 0.0    # USD per standard lot round trip
    swap_long:     float = -0.8   # pips per night for long positions
    swap_short:    float = 0.2    # pips per night for short positions
    slippage_pips: float = 0.5    # average execution slippage
    pip_size:      float = 0.0001 # EURUSD pip size

    @property
    def total_entry_cost_pips(self) -> float:
        """Total pip cost on trade entry"""
        return self.spread_pips + self.slippage_pips

    def swap_cost_pips(self, direction: str,
                       nights_held: int) -> float:
        """Total swap cost in pips for nights held"""
        rate = self.swap_long if direction == 'BUY' else self.swap_short
        return rate * nights_held

    def commission_cost(self, lot_size: float) -> float:
        """Commission in account currency"""
        return self.commission * lot_size

    def total_cost_pips(self, direction: str,
                        nights_held: int,
                        lot_size: float,
                        pip_value: float = 10.0) -> float:
        """
        Total cost in pips including all fees.
        Converts commission to pips using pip_value.
        """
        entry_cost  = self.total_entry_cost_pips
        swap_cost   = self.swap_cost_pips(direction, nights_held)
        comm_pips   = (self.commission_cost(lot_size) / pip_value
                       if pip_value > 0 else 0)
        return entry_cost + abs(swap_cost) + comm_pips


@dataclass
class BacktestTrade:
    entry_time:     datetime
    exit_time:      datetime
    direction:      str
    entry:          float
    stop_loss:      float
    take_profit:    float
    exit_price:     float
    pips_gross:     float    # pips before costs
    pips_net:       float    # pips after all costs
    cost_pips:      float    # total cost in pips
    nights_held:    int
    result:         str      # WIN / LOSS
    rr_ratio:       float
    reason:         str
    trend:          str
    confluences:    int
    lot_size:       float


@dataclass
class BacktestResult:
    trades:           list  = field(default_factory=list)
    equity_curve:     list  = field(default_factory=list)
    total_trades:     int   = 0
    wins:             int   = 0
    losses:           int   = 0
    gross_profit:     float = 0.0
    gross_loss:       float = 0.0
    total_costs:      float = 0.0   # total fees paid in CAD
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
        win_trades = [t for t in self.trades if t.result == 'WIN']
        if not win_trades:
            return 0.0
        return round(
            sum(t.pips_net for t in win_trades) / len(win_trades), 1
        )

    @property
    def avg_loss_pips(self) -> float:
        loss_trades = [t for t in self.trades if t.result == 'LOSS']
        if not loss_trades:
            return 0.0
        return round(
            sum(t.pips_net for t in loss_trades) / len(loss_trades), 1
        )

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
    costs:                TradingCosts  = None,
    profile_window:       int           = 500,
    warmup_bars:          int           = 500,
    pip_size:             float         = 0.0001,
    starting_balance:     float         = Config.ACCOUNT_BALANCE,
    risk_percent:         float         = Config.RISK_PERCENT,
    use_session_profiles: bool          = True,
    verbose:              bool          = False,
) -> BacktestResult:
    """
    Walk-forward backtester with realistic trading costs.

    For each bar after warmup:
    1. Build volume profile on preceding profile_window bars
    2. Generate signal through all 5 priority filters
    3. Apply entry costs (spread + slippage) to entry price
    4. Monitor trade bar by bar until TP or SL hit
    5. Apply swap for each night held
    6. Log net result after all costs
    """
    if costs is None:
        costs = TradingCosts()  # use defaults

    log.info(
        f"Trading costs loaded:\n"
        f"  Spread:     {costs.spread_pips} pips\n"
        f"  Slippage:   {costs.slippage_pips} pips\n"
        f"  Commission: ${costs.commission:.2f}/lot\n"
        f"  Swap long:  {costs.swap_long} pips/night\n"
        f"  Swap short: {costs.swap_short} pips/night"
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

    total_bars = len(df)
    start_bar  = max(warmup_bars, profile_window)

    log.info(
        f"Backtester starting  |  "
        f"Total bars: {total_bars:,}  |  "
        f"Testing from bar: {start_bar:,}  |  "
        f"Profile window: {profile_window} bars"
    )

    result.equity_curve.append(balance)

    for i in range(start_bar, total_bars):
        current_bar  = df.iloc[i]
        current_time = df.index[i]
        current_date = current_time.date()

        # reset daily trade counter
        if current_date != last_date:
            trades_today = 0
            last_date    = current_date

        # ── manage open trade ─────────────────────────────────────
        if in_trade and current_trade is not None:
            ct = current_trade

            # check if overnight (new calendar day)
            bar_date = current_time.date()
            if bar_date != ct.get('last_bar_date'):
                ct['nights_held'] = ct.get('nights_held', 0) + 1
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
                # gross pips (before costs)
                if ct['direction'] == 'BUY':
                    gross_pips = (exit_price - ct['actual_entry']) / pip_size
                else:
                    gross_pips = (ct['actual_entry'] - exit_price) / pip_size

                # total cost in pips
                nights  = ct.get('nights_held', 0)
                lot     = ct['lot_size']
                sl_pips = abs(ct['actual_entry'] - ct['stop_loss']) / pip_size
                risk_am = balance * (risk_percent / 100)
                pip_val = risk_am / sl_pips if sl_pips > 0 else 0

                cost_pips = costs.total_cost_pips(
                    ct['direction'], nights, lot, pip_val
                )

                # net pips after costs
                net_pips = gross_pips - cost_pips

                # PnL
                pnl        = net_pips * pip_val
                balance   += pnl
                result_str = 'WIN' if net_pips > 0 else 'LOSS'

                if pnl > 0:
                    result.wins         += 1
                    result.gross_profit += pnl
                else:
                    result.losses      += 1
                    result.gross_loss  += pnl

                result.total_costs += cost_pips * pip_val

                trade = BacktestTrade(
                    entry_time   = ct['entry_time'],
                    exit_time    = current_time,
                    direction    = ct['direction'],
                    entry        = ct['entry'],
                    stop_loss    = ct['stop_loss'],
                    take_profit  = ct['take_profit'],
                    exit_price   = round(exit_price, 5),
                    pips_gross   = round(gross_pips, 1),
                    pips_net     = round(net_pips, 1),
                    cost_pips    = round(cost_pips, 2),
                    nights_held  = nights,
                    result       = result_str,
                    rr_ratio     = ct['rr_ratio'],
                    reason       = ct['reason'],
                    trend        = ct['trend'],
                    confluences  = ct['confluences'],
                    lot_size     = lot,
                )
                result.trades.append(trade)
                result.total_trades += 1

                # drawdown
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
                        f"pnl: {pnl:+.2f}  "
                        f"bal: {balance:.2f}"
                    )

                in_trade      = False
                current_trade = None

            continue

        # ── skip if max trades reached today ──────────────────────
        if trades_today >= Config.MAX_TRADES_PER_DAY:
            continue

        # ── build rolling profile ──────────────────────────────────
        window_df = df.iloc[max(0, i - profile_window):i]

        try:
            levels = build_volume_profile(window_df)
        except Exception as e:
            if verbose:
                log.warning(f"Profile build failed at bar {i}: {e}")
            continue

        # ── session profiles ───────────────────────────────────────
        multi_levels = None
        if use_session_profiles:
            try:
                multi_levels = build_multi_session_levels(window_df)
            except Exception:
                pass

        # ── check signal ───────────────────────────────────────────
        signal_df = df.iloc[max(0, i - 300):i + 1]

        try:
            signal = generate_signal(
                df           = signal_df,
                levels       = levels,
                multi_levels = multi_levels,
            )
        except Exception as e:
            if verbose:
                log.warning(f"Signal error at bar {i}: {e}")
            continue

        if signal is None:
            continue

        # ── apply spread + slippage to actual entry ────────────────
        entry_cost = costs.total_entry_cost_pips * pip_size
        if signal.direction == 'BUY':
            actual_entry = signal.entry + entry_cost  # worse fill
        else:
            actual_entry = signal.entry - entry_cost  # worse fill

        # ── calculate lot size ─────────────────────────────────────
        sl_pips  = abs(signal.entry - signal.stop_loss) / pip_size
        lot_size = calculate_position_size(
            account_balance = balance,
            risk_percent    = risk_percent,
            stop_loss_pips  = sl_pips,
        )

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
            'confluences':   signal.confluences,
            'lot_size':      lot_size,
            'nights_held':   0,
        }

        if verbose:
            log.info(
                f"OPEN  {signal.direction:4}  "
                f"entry: {actual_entry:.5f}  "
                f"sl: {signal.stop_loss:.5f}  "
                f"tp: {signal.take_profit:.5f}  "
                f"spread cost: {costs.total_entry_cost_pips:.1f} pips"
            )

    result.final_balance = round(balance, 2)
    return result


def print_results(result: BacktestResult,
                  costs:  TradingCosts = None,
                  symbol: str = Config.SYMBOL):
    """Print a clean summary of backtest results"""
    if costs is None:
        costs = TradingCosts()

    divider = "=" * 52

    log.info(f"""
{divider}
  BACKTEST RESULTS — {symbol}
{divider}

  COSTS MODELLED
  ────────────────────────────────────────────
  Spread:           {costs.spread_pips} pips per trade
  Slippage:         {costs.slippage_pips} pips per trade
  Commission:       ${costs.commission:.2f} per lot
  Swap long:        {costs.swap_long} pips/night
  Swap short:       {costs.swap_short} pips/night
  Total avg cost:   {result.avg_cost_pips:.1f} pips per trade
  Total fees paid:  ${result.total_costs:.2f} CAD

  TRADES
  ────────────────────────────────────────────
  Total trades:     {result.total_trades:>8,}
  Wins:             {result.wins:>8,}  ({result.win_rate}%)
  Losses:           {result.losses:>8,}

  PERFORMANCE (after all costs)
  ────────────────────────────────────────────
  Starting balance: {result.starting_balance:>10.2f} CAD
  Final balance:    {result.final_balance:>10.2f} CAD
  Net profit:       {result.net_profit:>+10.2f} CAD  ({result.net_profit_pct:+.1f}%)
  Profit factor:    {result.profit_factor:>10.2f}  (need > 1.5)
  Sharpe ratio:     {result.sharpe_ratio:>10.2f}  (need > 1.0)

  RISK
  ────────────────────────────────────────────
  Max drawdown:     {result.max_drawdown:>10.2f} CAD  ({result.max_drawdown_pct:.1f}%)
  Avg win:          {result.avg_win_pips:>+10.1f} pips (net)
  Avg loss:         {result.avg_loss_pips:>+10.1f} pips (net)

  VERDICT
  ────────────────────────────────────────────
  {'✅ Profit factor PASS  (> 1.5)' if result.profit_factor >= 1.5 else '❌ Profit factor FAIL  (< 1.5)'}
  {'✅ Win rate PASS  (> 40%)' if result.win_rate >= 40 else '❌ Win rate FAIL  (< 40%)'}
  {'✅ Drawdown PASS  (< 20%)' if result.max_drawdown_pct < 20 else '❌ Drawdown FAIL  (> 20%)'}
  {'✅ Sharpe PASS  (> 1.0)' if result.sharpe_ratio >= 1.0 else '❌ Sharpe FAIL  (< 1.0)'}
{divider}
    """)


def save_trade_journal(result: BacktestResult,
                       filepath: str = "data/processed/trade_journal.csv"):
    """Save every trade to CSV for manual review"""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for t in result.trades:
        rows.append({
            'entry_time':   t.entry_time,
            'exit_time':    t.exit_time,
            'direction':    t.direction,
            'entry':        t.entry,
            'stop_loss':    t.stop_loss,
            'take_profit':  t.take_profit,
            'exit_price':   t.exit_price,
            'pips_gross':   t.pips_gross,
            'cost_pips':    t.cost_pips,
            'pips_net':     t.pips_net,
            'nights_held':  t.nights_held,
            'result':       t.result,
            'rr_ratio':     t.rr_ratio,
            'lot_size':     t.lot_size,
            'trend':        t.trend,
            'confluences':  t.confluences,
            'reason':       t.reason,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    log.info(f"Trade journal saved → {filepath}  ({len(df)} trades)")
    return df
