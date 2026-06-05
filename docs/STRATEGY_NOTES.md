# Strategy Notes — accumulated knowledge

> Portable project brain. Committed to git so context survives across machines
> (Mac dev ↔ Windows desktop). Updated 2026-06-05.

This is the "what we know and why" companion to the README. If you're picking this
project up on a new machine (or a fresh Claude Code session), read this first.

---

## 1. What the strategy actually is

**Volume-profile mean reversion in ranging markets.** It finds institutional price
levels (POC / HVN / value-area) and enters on rejection candles when price tests them
— but **only when the market is NEUTRAL (ranging).** In trends, these levels get blown
through. The regime filter is the whole game.

- Profile timeframe: **H1**, fixed **200 bins**, rolling **400-bar** window.
- Entries: **M30** (resampled from M15) rejection candles; H1 fallback.
- Regime: **H1 ADX < 25** AND **NEUTRAL** trend (EMA50/200 cross + price-vs-EMA200 +
  price structure; NEUTRAL = the votes disagree).

## 2. Current proven-best config (EURUSD)

| Param | Value | Note |
|---|---|---|
| PROFILE_BINS | 200 | fixed, pair-agnostic (tick-size rejected) |
| PROFILE_WINDOW | 400 | sweep-tuned (was 500) |
| VOLUME_SPIKE_MULT | 1.4 | sweep-tuned (was 1.2) — entry vol > 1.4× 20-bar avg |
| MAX_RR_RATIO | 2.0 | raising it is CATASTROPHIC (see below) |
| ADX_THRESHOLD | 25 | ranging only |
| MIN_CONFLUENCE | 3 | |
| ENTRY_WICK_RATIO | 1.8 | M30 rejection wick/body |

**EURUSD result (23yr, M30 entries, real costs): PF 2.24, +68.9%, 7.8% DD, Sharpe 6.03,
35 trades.** All tunable knobs live in `Config` ("Strategy mechanics" block).

## 3. What's been REJECTED — do NOT re-litigate (all data-disproven)

- **Tick-size bins** — PF 1.42 and 3-4× slower. Use fixed 200 bins.
- **Trend-following / counter-trend VP entries** — 34-41% win, PF < 0.85. NEUTRAL-only
  is load-bearing.
- **Widening the NEUTRAL gate** (swing-pivot structure, dropping EMA cross, ADX-only) —
  all blow up (−90% to −98%). The edge is the NARROWNESS of the ~3-5% NEUTRAL gate; the
  EMA cross is load-bearing precisely because its rare disagreement-with-price pinpoints
  exhaustion/reversion. Counterintuitive but emphatic.
- **Raising MAX_RR_RATIO** (2.5/3.0) — PF 1.23 / 0.68, huge drawdown. Price reverts to
  fair value then REVERSES; winners do not run. The ≤2.0 cap is the edge, not a limit.
- **Value-area edge fades** (VAH→SELL / VAL→BUY) — PF 1.83→1.61; a VA edge alone is
  < MIN_CONFLUENCE so it only admits marginal trades. Kept behind ENABLE_VA_EDGE_FADES
  (default off).
- **Conviction sizing by confluence count** — confluence count is INVERSELY correlated
  with win rate (3-conf 62% vs 4-conf 50%). Don't size up on it.

## 4. Multi-pair findings (Dukascopy real volume, 22yr, 2026-06)

First true multi-pair test on real tick volume. **The edge is concentrated, not portable:**

| Pair | PF (H1 entries) | Status |
|---|---|---|
| EURUSD | 1.88 | ✅ confirmed (London/NY) |
| USDJPY | 1.45 → **1.86** with Asian session | ✅ confirmed (needs Asian) |
| AUDUSD | 0.75 → 1.11 with Asian | 🟡 promising, not passing |
| GBPUSD / NZDUSD / USDCHF / USDCAD | 0.33–0.85 | ❌ no edge |

Key lessons:
- A GBPUSD "PF 2.18" from a 5-trade HistData test was a **mirage** (proxy volume, 3 good
  years). Real 22yr = 0.85. Never trust small favorable windows.
- **Session filter bias matters**: London/NY-only handicaps AUD/NZD/JPY (Tokyo
  liquidity). USDJPY flips fail→pass when allowed its Asian session
  (`INCLUDE_ASIAN_SESSION` / `run_multi_pair.py --asian`). Make this per-pair eventually.
- Frequency is gated by the regime/confluence stack (~0.5-1 trade/yr/pair), NOT sessions.
  Multi-pair is the only frequency lever — but only ~2-3 pairs actually carry the edge.

## 5. Data pipeline — how to rebuild `data/db/forex.db` on a new machine

The DuckDB file and raw data are **gitignored** (too big). On a fresh clone, rebuild:

- **Dukascopy (preferred — real tick volume):** needs Node.js. `dukascopy-node` via npx.
  ```
  python scripts/download_dukascopy.py --majors --from 2004-01-01
  ```
  H1 is fast (~74s/pair); M15 is slow (~38min/pair — Dukascopy serves sub-hourly as
  hourly artifacts). Use `--timeframes H1` first for a quick read.
- **HistData M1 CSVs (fallback, NO volume):** `scripts/import_histdata.py SYMBOL` —
  resamples M1→H1/M15 with a bar-count volume proxy. WARNING: the proxy saturates, so
  the volume-spike filter is effectively disabled. Only use if Dukascopy unavailable.
- **Legacy broker CSVs → `scripts/import_csv.py`** (the original EURUSD source).

Provider switch: `Config.DATA_SOURCE` / `get_provider()` → "offline" (DuckDB) or "mt5".
Backtests always run offline; MT5 is the live/fetch layer only.

## 6. Infra gotchas (learned the hard way)

- **Do NOT kill a process mid-write to DuckDB.** It corrupted `idx_ohlcv_main`
  ("Failed to delete all rows from index" + INTERNAL errors on filtered queries).
  Recovery: open RW, `DROP INDEX idx_ohlcv_main`, `CHECKPOINT` — table data survives.
- **dukascopy-node:** use `-fr` (skip failed artifact after retries); do NOT use `-re`
  (retry-on-empty) — it retries every empty weekend artifact and hangs M15 pulls.
- Backtests read the DB **read-only** (`get_connection(read_only=True)`) → many can run
  in parallel without lock contention. `scripts/sweep_run.py` exploits this.

## 7. Next steps

1. Download M15 for survivors (EURUSD, USDJPY, AUDUSD, +GBPUSD control) → proper M30 test.
2. Make Asian session a per-pair preset (JPY/AUD = Asian; EUR/GBP = London/NY).
3. Shared-timeline portfolio backtester (one equity curve, correlation-aware) — still TODO.
4. Time-based exit, sharpen the NEUTRAL gate, Kelly-aware risk — see ideas in commit history.

Re-validate any EURUSD-tuned parameter across pairs before trusting it.
