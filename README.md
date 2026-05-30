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
| Total trades | 31 | |
| Win rate | **58.1%** | > 40% |
| Profit factor | **1.81** | > 1.5 |
| Net profit | **+$2,397 CAD (+47.9%)** | > 0 |
| Max drawdown | **7.3%** | < 20% |
| Sharpe ratio | **4.87** | > 1.0 |

Costs modelled to match forex.com Standard Account: 1.2 pip spread, 0.5 pip slippage,
overnight swap (including triple swap Wednesdays), micro lot (0.01) minimum.

### Backtest Evolution

Every version of the strategy was tested on the full 23-year dataset. Failed experiments
are included — they show what was learned, not just what worked.

| Version | Trades | Win % | PF | Net P&L | Max DD | Outcome |
|---------|--------|-------|----|---------|--------|---------|
| H1 only (baseline) | 14 | 57.1% | 3.35 | +19.7% | 2.5% | Edge confirmed, too infrequent |
| M15 entries (1.5× wick) | 78 | 35.9% | 1.35 | +26.3% | 15.7% | Noise — M15 wick threshold too low |
| M15 entries (2.0× wick) | 51 | 43.1% | 1.91 | +40.6% | 7.6% | Solid — 3.6× frequency of H1 |
| **M30 entries (1.8× wick)** | **31** | **58.1%** | **1.81** | **+47.9%** | **3.3%** | **Current — best risk-adjusted** |
| Trend-following VP (tested) | 116 | 37.1% | 0.71 | −53.4% | 57.3% | Rejected — see findings below |

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
1. Price within 5 pips of H1 POC or non-POC HVN
2. H1 ADX < 25  (ranging market — trending markets invalidate VP levels)
3. H1 trend = NEUTRAL  (EMA50/200 + price vs EMA200 + price structure: ≤1/3 agree)
4. Rejection candle at level:
     M30 bar: lower/upper wick > body × 1.8, close in signal direction
     H1 fallback: wick > body × 1.5
5. Volume above 20-bar average at the level
6. Session POC confluence ≥ 2 timeframes (daily/weekly/monthly/long-term)
7. ≥ 3 independent confluences total
8. Stop loss ≥ 40% of ATR(14) — smaller stops get hit by noise
9. R:R ≥ 1.5:1
10. R:R capped at 2.0:1 (TP at nearest LVN — frequent small wins over rare large wins)
```

### Multi-Timeframe Design

```
H1  ── Volume profile building  (500-bar rolling window, 100 bins)
H1  ── Regime detection         (ADX + NEUTRAL trend filter)
H1  ── Session profiles         (daily/weekly/monthly POC confluence)
M30 ── Entry candles            (resampled from M15 data)
M30 ── SL placement             (wick low/high + 2 pips)
H1  ── TP placement             (nearest LVN in signal direction)
```

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

---

## Project Structure

```
forex-volume-profile/
├── scripts/
│   ├── import_csv.py            ← one-shot: imports broker CSVs into DuckDB
│   └── run_backtest.py          ← walk-forward backtest + equity curve chart
├── src/
│   ├── config.py                ← all parameters in one place
│   ├── backtester.py            ← MT5-realistic walk-forward simulator
│   ├── main.py                  ← live trading entry point
│   ├── data/
│   │   ├── base_provider.py     ← abstract data interface (Mac/Windows swap)
│   │   ├── csv_provider.py      ← Mac: DuckDB query
│   │   └── mt5_provider.py      ← Windows: MT5 API
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
| `POC_ZONE_PIPS` | 5 | H1 bins are ~4 pips wide — narrower misses entries |
| `HVN_THRESHOLD` | 0.70 | Bins with volume ≥ 70% of max bin count as HVN |
| `ADX_THRESHOLD` | 25.0 | Skip signals when market is trending |
| `MIN_STOP_ATR_MULT` | 0.40 | SL must be ≥ 40% of ATR(14) — avoids noise hits |
| `MIN_RR_RATIO` | 1.5 | Minimum reward-to-risk |
| `MAX_RR_RATIO` | 2.0 | Cap targets — larger R:R almost always hits SL |
| `MIN_CONFLUENCE` | 3 | ≥ 3 independent filters must agree |
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

**Phase 2 — Frequency (next)**
- [ ] Multi-pair expansion — GBPUSD, USDJPY, AUDUSD (same logic, more signals)
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

*EURUSD · 23-year walk-forward · 2003–2026 · Last updated May 2026*
