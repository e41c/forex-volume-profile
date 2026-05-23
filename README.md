# 🐲 Forex Volume Profile Bot

> Low risk. High discipline. Long term profits.*

A professional-grade algorithmic trading system built in Python, using volume profile
analysis to identify institutional price levels in the EURUSD forex market.
Developed on macOS M1, designed to deploy on a Windows VPS running MetaTrader 5.

---

## 🗺️ Project Vision

Most retail traders lose because they trade against institutions.
This bot finds where institutions **have already traded** — the high-volume price nodes —
and positions with them, not against them.

Volume profile is not a retail indicator. It is what prop desks, hedge funds,
and market makers use to understand where real value lives in the market.
We are building a system that sees what they see.

---

## ⚙️ Architecture Overview

```
Mac M1 (Development & Backtesting)
├── data/raw/          ← 15 years of broker EURUSD CSVs (gitignored)
├── data/db/forex.db   ← DuckDB database (gitignored)
└── src/               ← all strategy and analysis code

          │  git push / pull
          ▼

GitHub (Private Repository)
└── code only — no data, no secrets

          │  git pull
          ▼

Windows VPS (Live Execution — future)
├── MetaTrader5 running
└── Python bot connected via MT5 API
```

### Why This Architecture

- **Mac M1** does not support the MT5 Python API — so all development,
  backtesting and research happens here using historical CSV data
- **DuckDB** stores all historical data locally — no server required,
  single file, handles hundreds of millions of rows with ease
- **BaseDataProvider abstraction** means strategy code never changes
  between environments — only the data source swaps
- **Windows VPS** runs MT5 and the live bot — pulls the same codebase
  from GitHub, swaps to the MT5 provider automatically

---

## 📁 Project Structure

```
forex-volume-profile/
├── data/
│   ├── raw/                        ← broker CSVs by year (gitignored)
│   │   └── EURUSD_GMT+2_US-DST_YYYY/
│   │       ├── *_bars_M1.csv
│   │       ├── *_bars_M5.csv
│   │       ├── *_bars_M15.csv
│   │       ├── *_bars_M30.csv
│   │       ├── *_bars_H1.csv
│   │       ├── *_bars_H4.csv
│   │       ├── *_bars_D1.csv
│   │       ├── *_bars_W1.csv
│   │       └── *_bars_MN1.csv
│   ├── db/
│   │   └── forex.db                ← DuckDB (gitignored)
│   └── processed/                  ← charts, parquet cache (gitignored)
├── logs/                           ← runtime logs (gitignored)
├── notebooks/                      ← Jupyter research notebooks
├── scripts/
│   └── import_csv.py               ← one-shot CSV importer
├── src/
│   ├── config.py                   ← all parameters in one place
│   ├── main.py                     ← entry point
│   ├── visualizer.py               ← chart rendering
│   ├── data/
│   │   ├── base_provider.py        ← abstract data contract
│   │   ├── csv_provider.py         ← Mac: reads from DuckDB
│   │   └── mt5_provider.py         ← Windows: reads from MT5 API
│   ├── indicators/
│   │   └── volume_profile.py       ← POC, HVN, LVN detection
│   ├── strategy/
│   │   └── vp_strategy.py          ← signal generation, position sizing
│   └── utils/
│       └── logger.py               ← structured logging
├── tests/
│   └── test_volume_profile.py
├── .env.example                    ← MT5 credentials template
├── requirements.txt
└── README.md
```

---

## ✅ What Is Built — Phase 1 & 2 Complete

### Data Infrastructure
- [x] 15 years of EURUSD data (M1 through MN1) stored in DuckDB
- [x] Broker timezone conversion: GMT+2 US DST → UTC automatically
- [x] Smart CSV importer — skips already-imported years, handles all timeframes
- [x] MT5-compatible data format throughout — zero conversion needed at deployment
- [x] Data cached in parquet for fast repeated access

### Analysis Engine
- [x] Volume profile builder from OHLCV data
- [x] Point of Control (POC) — highest volume price level
- [x] High Volume Nodes (HVN) — institutional accumulation zones
- [x] Low Volume Nodes (LVN) — fast-move zones, used as targets
- [x] Multi-timeframe data access (any timeframe, any date range)

### Signal Framework
- [x] Candle rejection detection at POC and HVN levels
- [x] Bullish and bearish signal identification
- [x] Minimum R:R ratio filter (default 2:1)
- [x] 1% risk position size calculator
- [x] Max trades per day limit

### Visualizer
- [x] Dark-themed price chart with POC, HVN, LVN overlaid
- [x] Volume profile histogram with colour-coded levels
- [x] Chart saved as PNG automatically

### Developer Experience
- [x] Structured logging to file and terminal
- [x] Single config.py — all parameters in one place
- [x] OS auto-detection — Mac uses CSV/DuckDB, Windows uses MT5
- [x] .env for secrets — never hardcoded, never pushed to GitHub
- [x] Full .gitignore — data, secrets, logs all protected

---

## 🔮 Roadmap — What We Are Building Next

### Phase 3 — Backtesting Engine
- [ ] Walk-forward backtester across all 15 years of data
- [ ] Performance metrics: win rate, profit factor, max drawdown, Sharpe ratio
- [ ] Trade journal export to CSV — every signal logged
- [ ] Parameter optimisation — find best BINS, HVN threshold, pip zones
- [ ] Equity curve visualisation

### Phase 4 — Strategy Refinement
- [ ] Rolling volume profile (session-based, not full-history)
- [ ] Multi-timeframe confluence — H1 profile + M15 entry
- [ ] News filter — skip trading around high-impact events
- [ ] Session filter — only trade London and New York sessions
- [ ] Spread filter — skip when spread exceeds threshold

### Phase 5 — MT5 Integration (Windows VPS)
- [ ] MT5 provider fully wired and tested on demo account
- [ ] Automated order placement with stop loss and take profit
- [ ] Trailing stop to breakeven at 1:1 R
- [ ] Position monitoring and partial close logic
- [ ] Telegram alerts — signal fired, trade opened, trade closed

### Phase 6 — VPS Deployment
- [ ] Windows VPS setup and hardening
- [ ] Bot runs as a scheduled task — 24 hours, 5 days a week
- [ ] Health monitoring — alert if bot goes silent
- [ ] Weekly performance report generated automatically
- [ ] Remote log access

### Phase 7 — Live Trading
- [ ] Minimum 3 months profitable on demo before going live
- [ ] $1000 CAD starting capital
- [ ] 1% risk per trade — maximum $10 CAD risk at entry
- [ ] Monthly performance review and parameter adjustment
- [ ] Scale account size only after consistent profitability

---

## 🐲 Goblin Trading Rules — Non-Negotiable

These rules exist because discipline is the only edge that cannot be back-tested away.

```
1. Never risk more than 1% per trade
2. Never trade more than 3 times per day
3. Never move a stop loss further away — only to breakeven or closer
4. Never trade during major news events
5. Demo trade for minimum 3 months before any real money
6. If the bot has 3 consecutive losses — pause and review
7. The bot never overrides these rules — ever
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- macOS M1 (development) or Windows (live execution)

### Setup

```bash
# clone
git clone https://github.com/YOURUSERNAME/forex-volume-profile.git
cd forex-volume-profile

# virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac
# venv\Scripts\activate           # Windows

# dependencies
pip install -r requirements.txt

# environment variables
cp .env.example .env
# edit .env with your MT5 credentials (Windows only)
```

### Import Historical Data (Mac)

```bash
# drop your year folders into data/raw/ then:
python scripts/import_csv.py
```

### Run The Bot

```bash
python -m src.main
```

---

## 📦 Dependencies

```
pandas          — data manipulation
numpy           — numerical operations
duckdb          — local time-series database
pytz            — timezone conversion
matplotlib      — charting
mplfinance      — candlestick charts
python-dotenv   — environment variable management
pytest          — testing
pyarrow         — parquet file support
MetaTrader5     — MT5 API (Windows only)
```

---

## 🔐 Security

- `.env` is gitignored — MT5 credentials never leave your machine
- `data/` is gitignored — historical data stays local
- `logs/` is gitignored — runtime logs stay local
- Only source code is pushed to GitHub

---

## 📊 Data Format

All data is stored and processed in **MT5-compatible UTC format**:

| Column | Type | Description |
|--------|------|-------------|
| time | TIMESTAMPTZ | UTC timestamp (converted from broker GMT+2 US DST) |
| open | DOUBLE | Bar open price |
| high | DOUBLE | Bar high price |
| low | DOUBLE | Bar low price |
| close | DOUBLE | Bar close price |
| volume | BIGINT | Tick volume |

Source data: EURUSD, broker GMT+2 with US DST rules, 2003–present
Timeframes available: M1, M5, M15, M30, H1, H4, D1, W1, MN1

---

## 🤝 About This Project

Built by two goblins who believe retail traders deserve institutional tools.

> *"The market is a device for transferring money from the impatient to the patient."*
> — Warren Buffett, honorary goblin

---

*Last updated: May 2026*
