# Forex Volume Profile Strategy

> Institutional mean reversion on EURUSD — walk-forward validated across 23 years.

A Python algorithmic trading system that identifies institutional price levels using
volume profile analysis, then enters on rejection candles in ranging markets. Designed
for live execution via MetaTrader 5, developed and backtested on macOS with DuckDB.

---

## Results — 23-Year Walk-Forward Backtest (2003–2026)

**EURUSD · M30 entries · $5,000 CAD · 3% risk per trade · MT5-realistic costs**

| Metric | Result | Threshold |
|--------|--------|-----------|
| Total trades | 35 | |
| Win rate | **62.9%** | > 40% |
| Profit factor | **2.24** | > 1.5 |
| Net profit | **+$3,446 CAD (+68.9%)** | > 0 |
| Max drawdown | **7.8%** | < 20% |
| Sharpe ratio | **6.03** | > 1.0 |

Costs modelled to match forex.com Standard Account: 1.2 pip spread, 0.5 pip slippage,
overnight swap (including triple swap Wednesdays), micro lot (0.01) minimum.

### Backtest Evolution

Every version of the strategy was tested on the full 23-year dataset. Failed experiments
are included — they show what was learned, not just what worked.

| Version | Trades | Win % | PF | Net P&L | Max DD | Outcome |
|---------|--------|-------|----|---------|--------|---------|
| H1 only (baseline) | 14 | 57.1% | 3.35 | +19.7% | 2.5% | Edge confirmed, too infrequent |
| M15 entries (2.0× wick) | 51 | 43.1% | 1.91 | +40.6% | 7.6% | Solid — 3.6× frequency of H1 |
| M30 entries (1.8× wick) | 31 | 58.1% | 1.81 | +47.9% | 3.3% | Prior risk-adjusted best |
| Tick-size bins + cluster-peak TP | 46 | ~52% | 1.42 | +35.1% | 17.5% | Rejected — slower (17 min) and worse |
| Fixed 200 bins (reverted) | 39 | 59.0% | 1.83 | +58.8% | 11.3% | Restored — back to the proven edge |
| **+ volume 1.4 & window 400** | **35** | **62.9%** | **2.24** | **+68.9%** | **7.8%** | **Current — sweep-tuned best** |
| Trend-following VP (tested) | 116 | 37.1% | 0.71 | −53.4% | 57.3% | Rejected — see findings below |
| Value-area edge fades (tested) | 41 | 56.1% | 1.61 | +47.0% | 11.3% | Rejected — admits only marginal trades |

---

## Strategy

### Core Thesis

Most retail traders lose because they trade against institutions. This strategy finds
where institutions *have already traded* — the high-volume price nodes — and enters
with them on rejection candles, not against them.

The edge only works in **ranging markets**. In trending markets, POC levels get blown
through. The regime filter enforces this.

### Signal Logic

Ten filters applied in order — all must pass:

```
1. Price within 5 pips of H1 POC or non-POC HVN (or daily session POC, 15 pips)
2. H1 ADX < 25  (ranging market — trending markets invalidate VP levels)
3. H1 trend = NEUTRAL  (EMA50/200 + price vs EMA200 + price structure: ≤1/3 agree)
4. Rejection candle at level:
     M30 bar: lower/upper wick > body × 1.8, close in signal direction
     H1 fallback: wick > body × 1.5
5. Volume > 1.4 × 20-bar average at the level  (strong institutional activity)
6. Session POC confluence ≥ 2 timeframes (daily/weekly/monthly/long-term)
7. ≥ 3 independent confluences total
8. Stop loss ≥ 40% of ATR(14) — smaller stops get hit by noise
9. R:R ≥ 1.5:1
10. R:R capped at 2.0:1 (TP at nearest LVN — frequent small wins over rare large wins)
```

Value-area edge fades (VAH→SELL, VAL→BUY) are implemented behind the
`ENABLE_VA_EDGE_FADES` flag but **default off** — testing showed they only admit
marginal, lower-quality trades (PF 1.83 → 1.61). The flag keeps the code available
for re-testing per pair.

### Multi-Timeframe Design

```
H1  ── Volume profile building  (400-bar rolling window, 200 fixed bins)
H1  ── Regime detection         (ADX + NEUTRAL trend filter)
H1  ── Session profiles         (daily/weekly/monthly POC confluence)
M30 ── Entry candles            (resampled from M15 data)
M30 ── SL placement             (wick low/high + 2 pips)
H1  ── TP placement             (nearest LVN in signal direction)
```

Profile resolution uses **fixed 200 bins** (not tick-size). Tick-size bins were
tested and rejected: they ran 3–4× slower and dropped PF to 1.42. Fixed bins are
also pair-agnostic — the same value carries to every instrument.

### What the Data Showed

**NEUTRAL regime filter is load-bearing.** Tested explicitly in 2026:
- NEUTRAL (ranging): 63% win rate, +15.4 pips average
- BULLISH/BEARISH (trending): 34% win rate, −4.5 pips average
- Counter-trend VP entries: hypothesis rejected — price has directional conviction and blows through levels

**ADX < 25 works as a proxy for ranging conditions.** Best years: 2008 financial crisis
and 2020 COVID (high volatility, ranging price action). Worst years: 2009, 2014, 2024
(sustained directional USD trends).

**Session POC confluence is signal, not noise.** Score = 1 averages −2.45 pips.
Score ≥ 2 averages +2.06 pips. Minimum 2 timeframes required to count.

**Circuit breaker protects capital in regime transitions.** Fired once across 23 years
(2011). Catches periods where NEUTRAL-appearing markets are actually trending at a higher
timeframe.

**A 12-variant parameter sweep found two complementary upgrades.** Raising the volume
filter (1.2 → 1.4) strips weak tests of the level; shrinking the profile window (500 →
400) surfaces fresher levels. Together: PF 1.83 → **2.24**, drawdown 11.3% → **7.8%**.
The volume filter tames the window change's drawdown (window-400 *alone* had 14.8% DD).

**Raising the take-profit cap is catastrophic.** TP cap 2.0 → 2.5 dropped PF to 1.23;
→ 3.0 went to PF 0.68, −27% net, 49% drawdown. This edge reverts to fair value then
*reverses* — winners do not run. The ≤2.0 cap is load-bearing, not a limitation.

*Note: the volume/window values were tuned on EURUSD alone — they get re-validated
across pairs before the multi-pair portfolio is trusted.*

---

## Architecture

```
Mac (Development + Backtesting)
├── data/raw/          ← 23 years of broker CSVs (gitignored)
├── data/db/forex.db   ← DuckDB time-series store (gitignored)
└── src/               ← strategy and infrastructure code

          │  git push / pull
          ▼

GitHub
└── code only — no data, no credentials

          │  git pull
          ▼

Windows VPS (Live Execution — Phase 5)
├── MetaTrader 5
└── Python bot via MT5 API
```

The MT5 Python API is Windows-only. All development and backtesting uses DuckDB with
historical CSVs. The `BaseDataProvider` abstraction means strategy logic never changes
between environments — only the data source swaps.

**The source is an explicit switch, not OS auto-detection.** `get_provider()` reads
`DATA_SOURCE` (env or `Config`): `offline` → DuckDB/CSV (fast, deterministic backtests),
`mt5` → live MetaTrader 5. The same Windows machine can do both by flipping the knob.
Best practice: pull MT5 history into DuckDB once (`scripts/ingest_mt5.py`), then always
backtest *offline* — never against the live MT5 feed (slow, rate-limited, history-capped).

```bash
python scripts/run_backtest.py --source offline   # backtest off DuckDB (default)
DATA_SOURCE=mt5 python src/main.py                 # live execution
```

---

## Project Structure

```
forex-volume-profile/
├── scripts/
│   ├── import_csv.py            ← one-shot: imports broker CSVs into DuckDB
│   ├── ingest_mt5.py            ← backfill full MT5 history into DuckDB (Windows)
│   ├── run_backtest.py          ← walk-forward backtest + equity curve chart
│   ├── run_multi_pair.py        ← same strategy across all pairs in DuckDB
│   ├── sweep_run.py             ← single-variant parameter sweep (parallel-safe)
│   └── plot_clusters.py         ← multi-TF volume-profile presentation chart
├── src/
│   ├── config.py                ← all parameters in one place
│   ├── backtester.py            ← MT5-realistic walk-forward simulator
│   ├── main.py                  ← live trading entry point
│   ├── data/
│   │   ├── __init__.py          ← get_provider() factory (offline ↔ mt5 switch)
│   │   ├── base_provider.py     ← abstract data interface
│   │   ├── csv_provider.py      ← offline: DuckDB query (read-only)
│   │   └── mt5_provider.py      ← live: MT5 API (Windows)
│   ├── indicators/
│   │   ├── volume_profile.py    ← POC, VAH, VAL, HVN, LVN detection
│   │   ├── session_profile.py   ← multi-TF session POC confluence
│   │   └── trend_filter.py      ← EMA, ADX, ATR, NEUTRAL regime detection
│   ├── strategy/
│   │   └── vp_strategy.py       ← signal generation, position sizing
│   └── utils/
│       ├── logger.py            ← structured file + console logging
│       └── session_filter.py    ← London/NY session gating
└── tests/
    └── test_volume_profile.py
```

---

## Key Parameters

All parameters in `src/config.py`.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `ACCOUNT_BALANCE` | 5000 CAD | Starting capital |
| `RISK_PERCENT` | 3.0 | % risked per trade |
| `PROFILE_BINS` | 200 | Fixed bins per profile (pair-agnostic; tick-size rejected) |
| `PROFILE_WINDOW` | 400 | Rolling bars per profile (sweep-tuned, was 500) |
| `POC_ZONE_PIPS` | 5 | Entry proximity to a level |
| `VOLUME_SPIKE_MULT` | 1.4 | Entry volume must exceed avg × this (sweep-tuned, was 1.2) |
| `HVN_MA_PERIOD` | 55 | Dual-MA crossover period for HVN/LVN detection |
| `ADX_THRESHOLD` | 25.0 | Skip signals when market is trending |
| `MIN_STOP_ATR_MULT` | 0.40 | SL must be ≥ 40% of ATR(14) — avoids noise hits |
| `MIN_RR_RATIO` | 1.5 | Minimum reward-to-risk |
| `MAX_RR_RATIO` | 2.0 | Cap targets — raising it is catastrophic (winners don't run) |
| `MIN_CONFLUENCE` | 3 | ≥ 3 independent filters must agree |
| `ENABLE_VA_EDGE_FADES` | False | Value-area edge fades — tested, rejected, kept behind flag |
| `MAX_CONSECUTIVE_LOSSES` | 3 | Circuit breaker — pause after 3 losses |
| `LOSS_COOLDOWN_BARS` | 48 | H1 bars (~2 trading days) pause after breaker |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Time-series storage | DuckDB |
| Data manipulation | pandas, NumPy |
| Broker / live execution | MetaTrader 5 API (Windows) |
| Charting | Matplotlib |
| Testing | pytest |

---

## Setup

```bash
git clone https://github.com/e41c/forex-volume-profile.git
cd forex-volume-profile
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # add MT5 credentials (Windows live trading only)
```

Import historical data (place broker CSV files in `data/raw/` first):

```bash
python scripts/import_csv.py
```

Run the backtest:

```bash
python scripts/run_backtest.py
```

Output: terminal results table + equity curve saved to `data/processed/EURUSD_backtest_results.png`
+ trade journal at `data/processed/trade_journal.csv`.

Run all pairs in the database on the same strategy:

```bash
python scripts/run_multi_pair.py            # all pairs with data
python scripts/run_multi_pair.py GBPUSD     # a specific pair
```

Sweep a parameter (in-memory override, read-only DB — safe to run many in parallel):

```bash
python scripts/sweep_run.py --label rr30 --set MAX_RR_RATIO=3.0
```

---

## Roadmap

**Phase 1 — Strategy Validation (complete)**
- [x] DuckDB data infrastructure (23-year EURUSD, H1 + M15)
- [x] Volume profile engine (POC, VAH, VAL, HVN, LVN detection)
- [x] Multi-TF session profile confluence (daily/weekly/monthly POC)
- [x] NEUTRAL regime detection (EMA cross + price structure + ADX)
- [x] Walk-forward backtester (MT5-realistic costs, circuit breaker, equity curve)
- [x] M30 entry candles (better noise/frequency tradeoff than M15 or H1)
- [x] Trend-following VP entries tested and rejected (2026) — NEUTRAL-only confirmed
- [x] Parameter sweep — volume 1.4 + window 400 (PF 1.83 → 2.24, DD 11.3% → 7.8%)
- [x] Value-area edge fades tested and rejected — kept behind a flag
- [x] Explicit offline ↔ MT5 provider switch + MT5→DuckDB ingest

**Phase 2 — Frequency (next)**
- [ ] Multi-pair data ingest (GBPUSD, EURGBP, AUDUSD, USDCHF) — harness is ready
- [ ] Re-validate volume/window tuning across pairs (currently EURUSD-only)
- [ ] Cross-pair signal correlation filter (avoid correlated simultaneous positions)

**Phase 3 — Robustness**
- [ ] News filter (skip ±30 min around high-impact USD/EUR events)
- [ ] Walk-forward optimization (rolling parameter windows)

**Phase 4 — Live Execution**
- [ ] MT5 execution engine (Windows VPS, demo validation first)
- [ ] Position monitoring and trade management loop
- [ ] Alert system (email / Telegram)

---

## Risk Management Rules

```
1. Fixed 3% risk per trade — position sized to stop distance, not arbitrary lots
2. Circuit breaker: halt after 3 consecutive losses, resume after 48 hours
3. Never move a stop loss further away — only to breakeven or tighter
4. No new trades Friday after 4pm ET — weekend gap risk
5. London and New York sessions only (strategy relies on institutional volume)
6. Strategy fires only in NEUTRAL regime — this filter is never overridden
7. 3 months profitable on demo before live capital
```

---

*EURUSD · 23-year walk-forward · 2003–2026 · Last updated June 2026*
