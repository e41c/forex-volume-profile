# Forex Volume Profile Bot

> Low risk. High discipline. Institutional-grade mean reversion.

A Python algorithmic trading system using volume profile analysis to identify institutional
price levels in the EURUSD forex market. Built on macOS, designed to deploy on Windows VPS
via MetaTrader 5.

---

## Strategy Summary

Most retail traders lose because they trade against institutions.
This bot finds where institutions **have already traded** — the high-volume price nodes —
and positions with them at rejection candles, not against them.

**Core edge:** Volume profile mean reversion at Point of Control (POC) and High Volume Node
(HVN) levels. This edge only works in **ranging markets** — in trends, POC levels get blown
through. The strategy therefore requires all three to be true before firing a signal:

```
1. Price at an institutional level (POC or HVN cluster)
2. Ranging market (ADX < 25 — no strong trend)
3. H1 trend regime is NEUTRAL (EMA50/200 + price structure + higher lows/highs all disagree)
```

All three together = institutional level + no trend fighting us = mean reversion edge.

---

## Backtest Results (23-Year Walk-Forward, 2003–2026)

EURUSD H1 volume profile, M15 entry candles, MT5-realistic costs (spread + slippage + swap).

| Version | Trades | Win Rate | Profit Factor | Net Profit | Max DD | Notes |
|---------|--------|----------|---------------|------------|--------|-------|
| H1 only (baseline) | 14 | 57.1% | 3.35 | +19.7% | 2.5% | Too infrequent |
| M15 entry (1.5× wick) | 78 | 35.9% | 1.35 | +26.3% | 15.7% | Too noisy |
| **M15 entry (2.0× wick)** | **51** | **43.1%** | **1.91** | **+40.6%** | **7.6%** | **Current — all pass** |

**Current strategy verdict: all 4 criteria pass**
- Profit factor: 1.91 (need > 1.5)
- Win rate: 43.1% (need > 40%)
- Max drawdown: 7.6% (need < 20%)
- Sharpe ratio: 4.73 (need > 1.0)

Costs modelled to match forex.com Standard Account: 1.2 pip spread, 0.5 pip slippage,
swap overnight, triple swap Wednesday, micro lot minimum, CAD account pip value.

---

## Architecture

```
Mac (Development & Backtesting)
├── data/raw/          ← 23 years of EURUSD CSVs (gitignored)
├── data/db/forex.db   ← DuckDB database (gitignored)
└── src/               ← strategy and analysis code

          │  git push / pull
          ▼

GitHub (Private Repository)
└── code only — no data, no secrets

          │  git pull
          ▼

Windows VPS (Live Execution — Phase 5)
├── MetaTrader5 running
└── Python bot connected via MT5 API
```

Mac does not support the MT5 Python API — all development and backtesting uses DuckDB
with historical CSVs. The `BaseDataProvider` abstraction means strategy code never
changes between environments — only the data source swaps.

---

## Multi-Timeframe Design

```
H1  — Volume profile (which levels matter)
H1  — Regime detection (ADX + NEUTRAL trend filter)
M15 — Entry candles (rejection pattern at H1 levels, tighter SL)
M15 — Trade management (TP/SL checked each bar)
```

The H1 timeframe is wide enough for the volume profile to capture meaningful institutional
activity. M15 sub-bars within each NEUTRAL H1 bar give 4× more entry opportunities without
loosening the regime filter. M15 wicks give tighter stops.

---

## Signal Logic (in order)

```
1. Price within 5 pips of H1 POC or non-POC HVN
2. H1 ADX < 25  (ranging market only — trending markets blow through levels)
3. H1 trend = NEUTRAL  (EMA50/200 crossover + price vs EMA200 + price structure)
4. Rejection candle at level:
     H1 bar: wick > body × 1.5
     M15 bar: wick > body × 2.0  (noisier timeframe needs stricter filter)
5. Volume above 20-bar average at that timeframe
6. Session POC confluence ≥ 2 timeframes (weekly/monthly/daily POCs near price)
7. Minimum 3 confluences total
8. Stop loss ≥ 40% of H1 ATR(14) — stops smaller than this get hit by noise
9. R:R between 2:1 and 4:1 — TP at nearest LVN above/below entry
```

---

## Project Structure

```
forex-volume-profile/
├── data/
│   ├── raw/                        ← broker CSVs by year (gitignored)
│   ├── db/forex.db                 ← DuckDB (gitignored)
│   └── processed/                  ← charts, trade journal (gitignored)
├── logs/                           ← runtime logs (gitignored)
├── notebooks/                      ← Jupyter research notebooks
│   └── 01_strategy_explorer.ipynb  ← interactive volume profile + signal walkthrough
├── scripts/
│   ├── import_csv.py               ← one-shot CSV importer
│   └── run_backtest.py             ← full walk-forward backtest + chart
├── src/
│   ├── config.py                   ← all parameters in one place
│   ├── backtester.py               ← MT5-realistic walk-forward backtester
│   ├── main.py                     ← live trading entry point (future)
│   ├── data/
│   │   ├── base_provider.py        ← abstract data contract
│   │   ├── csv_provider.py         ← Mac: reads from DuckDB
│   │   └── mt5_provider.py         ← Windows: reads from MT5 API
│   ├── indicators/
│   │   ├── volume_profile.py       ← POC, HVN, LVN detection
│   │   ├── session_profile.py      ← multi-TF session POC confluence
│   │   └── trend_filter.py         ← EMA, ADX, ATR, regime detection
│   ├── strategy/
│   │   └── vp_strategy.py          ← signal generation, position sizing
│   └── utils/
│       ├── logger.py               ← structured logging
│       └── session_filter.py       ← London/NY session filter
└── tests/
    └── test_volume_profile.py
```

---

## Key Parameters (src/config.py)

| Parameter | Value | Why |
|-----------|-------|-----|
| `POC_ZONE_PIPS` | 5 | H1 bins are ~4 pips wide — 3 was too tight |
| `HVN_THRESHOLD` | 0.70 | Bins with volume ≥ 70% of max bin |
| `ADX_THRESHOLD` | 25.0 | Skip signals when market is trending |
| `MIN_STOP_ATR_MULT` | 0.40 | SL must be ≥ 40% of ATR(14) to avoid noise hits |
| `MIN_RR_RATIO` | 2.0 | Minimum 2:1 reward-to-risk |
| `MAX_RR_RATIO` | 4.0 | Cap targets — R:R > 4 almost always hits SL not TP |
| `MIN_CONFLUENCE` | 3 | Need ≥ 3 independent filters confirming the level |
| `MAX_CONSECUTIVE_LOSSES` | 3 | Circuit breaker — pause after 3 losses in a row |
| `LOSS_COOLDOWN_BARS` | 48 | ~2 trading days pause after circuit breaker fires |

---

## What Works (Data-Backed)

- **NEUTRAL regime only** — BULLISH/BEARISH trend states lose (POC levels get blown through
  in trending markets). NEUTRAL = 57% win rate. Trending = 24-28% win rate.
- **ADX < 25 filter** — removes trending periods. Best years were 2008 crisis and 2020 COVID
  (ranging volatility). Worst years were 2014 and 2024 (sustained directional trends).
- **Session POC confluence ≥ 2** — score = 1 averages −2.45 pips. Score ≥ 2 averages
  +2.06 pips. Single TF alignment is noise; multi-TF clustering is signal.
- **ATR minimum stop** — stops < 40% of ATR(14) get hit by random noise.
  Eliminated many small-SL losing trades.
- **Circuit breaker** — protects capital during regime transitions. Fired 5 times across
  23 years at known bad periods: 2011, 2015, 2018, 2022, 2023.
- **M15 entries (2.0× wick)** — 3.6× more signals than H1 alone at similar quality.
  Requires 2.0× wick/body ratio (vs 1.5× for H1) to filter out M15 noise.

---

## Running the Backtest

```bash
source venv/bin/activate
python scripts/run_backtest.py
```

Output: terminal results + equity curve chart at `data/processed/EURUSD_backtest_results.png`
+ trade journal at `data/processed/trade_journal.csv`.

---

## Jupyter Notebooks (Strategy Explorer)

For interactive exploration, signal visualization, and parameter tuning:

```bash
source venv/bin/activate
pip install jupyter
jupyter notebook notebooks/01_strategy_explorer.ipynb
```

The notebook covers:
- Volume profile visualization (price chart + histogram)
- Signal walkthrough on specific dates
- Trade journal analysis by year, direction, confluence
- Equity curve comparison across parameter settings

---

## Setup

```bash
git clone https://github.com/e41c/forex-volume-profile.git
cd forex-volume-profile
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add MT5 credentials (Windows only)
```

Import historical data (drop year folders into `data/raw/` first):
```bash
python scripts/import_csv.py
```

---

## Goblin Trading Rules

Discipline is the only edge that cannot be backtested away.

```
1. Never risk more than 1% per trade
2. Circuit breaker: stop after 3 consecutive losses, pause 48 hours
3. Never move a stop loss further away — only to breakeven or closer
4. Never trade during major news events
5. 3 months profitable on demo before any real money
6. Strategy only fires in NEUTRAL regime — never override this
7. Scale lot size up only after 6 consecutive profitable months
```

---

## Roadmap

- [x] Data infrastructure (DuckDB, 23-year EURUSD, all timeframes)
- [x] Volume profile engine (POC, HVN, LVN)
- [x] Multi-timeframe session profiles (daily/weekly/monthly POC confluence)
- [x] Trend filter (EMA, ADX, ATR, NEUTRAL regime detection)
- [x] Walk-forward backtester (MT5-realistic costs, circuit breaker, equity curve)
- [x] M15 entry candles at H1 levels (3.6× more signals, regime unchanged)
- [ ] M30 entry timeframe experiment (less noise than M15, more frequent than H1)
- [ ] Lower R:R target (1.5:1) for more consistent small wins
- [ ] Multi-pair expansion (GBPUSD, USDJPY — same logic, more opportunities)
- [ ] News filter (skip ±30 min around high-impact events)
- [ ] MT5 live execution (Windows VPS, demo first)
- [ ] Telegram alerts

---

*Built by two goblins. Last updated: May 2026.*
