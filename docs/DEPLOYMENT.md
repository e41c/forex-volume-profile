# Deployment Roadmap — from backtest to funded live trading

The research is done: a validated, risk-managed, fundable multi-edge system. This is the
honest, concrete path from a backtest to real income — and the disciplines that keep it
trustworthy along the way.

> **Guiding principle:** prove it small and safe, get funded, scale the *capital* — never
> the leverage. Income = (edge quality) × (capital). The system runs identically on $5k or
> $1M; the goal is for the $1M to be a prop firm's.

---

## Phase 0 — where we are ✅

- 11 validated edges (out-of-sample + parameter-stability tested), combined on one
  risk-managed equity curve: ~3%/yr at ~8.8% max drawdown, Sharpe ~1.0, ~124 trades/yr.
- All offline/backtested on Dukascopy data via DuckDB. Reproducible.

---

## Phase 1 — live infrastructure (the engineering)

Move from "backtest that proves an edge" to "system that places orders." On the Windows
desktop with MetaTrader 5 (the codebase already has the `get_provider()` offline↔MT5 switch).

- [ ] **Broker / instrument coverage.** Confirm one broker (or a small set) offers the
      instruments we trade: 7 FX majors + ~21 crosses, the equity indices (S&P, Nasdaq, Dow,
      DAX, Nikkei), metals (gold, silver), energy (Brent). *If coverage is partial, deploy the
      covered subset first* — the system degrades gracefully (each edge is independent).
- [ ] **Live data feed.** `DATA_SOURCE=mt5` → pull recent bars (H1/daily) for every
      instrument each cycle. Keep ingesting into DuckDB so live and research share one store.
- [ ] **Signal engine.** A scheduled job (e.g. once per hour / once daily at the bar close)
      that rebuilds each edge's current signal from the latest bars and emits target positions.
- [ ] **Risk layer, live.** Port the portfolio sizing (vol-parity + per-sleeve budgeting +
      de-risk-in-drawdown + vol-target) to size live orders from current account equity.
- [ ] **Execution module.** Translate target positions → MT5 orders (entries, exits, stops),
      with idempotency (don't double-fill) and reconciliation against actual broker positions.
- [ ] **Monitoring + kill-switch.** Logging, daily P&L/exposure reports, alerts (email/
      Telegram), and a hard stop that flattens everything if drawdown or a data fault triggers.

---

## Phase 2 — demo forward-test (prove it live, risk-free)

- [ ] Run the full system on a **demo account** for **2–3 months minimum**.
- [ ] Compare live vs backtest: realised drawdown, trade frequency, and slippage. Confirm the
      live edge matches the simulated one (the honest check that the backtest wasn't fiction).
- [ ] Tune execution (order types, slippage assumptions) to reality. Fix any data/timing bugs.
- [ ] **Do not skip this.** A backtest is a hypothesis; the demo is the experiment.

---

## Phase 3 — prop-firm evaluation (get the capital)

The system's profile — **low, controlled drawdown and consistency** — is exactly what funded
programs reward (they care about risk discipline far more than flashy returns).

- [ ] **Choose a reputable firm.** Prioritise established, well-reviewed programs; avoid
      "challenge-fee farming" outfits. Read the rules carefully.
- [ ] **Map the rules → our risk layer.** Each firm sets: max daily drawdown, max total
      drawdown, profit target, minimum trading days, allowed instruments/leverage. Configure
      `--target-dd` and the sizing comfortably *inside* their daily and total limits (e.g.
      target 5–6% so a bad day never breaches a 5% daily cap).
- [ ] **Pass the evaluation** with the same system, unchanged. Consistency is the proof.

---

## Phase 4 — funded & scale (the income)

- [ ] Trade the funded account live, withdraw profits on schedule, keep meticulous records.
- [ ] **Scale capital, not risk:** add accounts / larger allocations / multiple firms. The
      same % return on $200k → $1M multiplies income with no change to the strategy.
- [ ] Re-validate the edges periodically (markets evolve); retire any that decay (we have the
      gauntlet to catch it), and keep hunting new uncorrelated edges to lift the ceiling.

---

## Cross-cutting disciplines (what keeps it trustworthy)

- **Risk first, always.** The drawdown limit is sacred; never override it for a "feeling."
- **Records everything.** A clean trade log and equity history is both an edge (you learn)
  and a credibility asset (you can prove your track record).
- **Honesty about regime.** A great month is not skill; a bad month is not failure. Judge the
  system over quarters and years, against its backtested expectation.
- **One change at a time.** Any future change is re-validated through the gauntlet before it
  touches live capital.

---

## Realistic economics (no hype)

At the safe ~9% drawdown setting (~3%/yr long-run; recent years have run hotter but that is
not the baseline):

| Capital | ~Annual (3%) | Your 80% split |
|---------|--------------|----------------|
| $5,000 (own) | ~$150 | — |
| $200,000 (funded) | ~$6,000 | ~$4,800 |
| $1,000,000 (scaled) | ~$30,000 | ~$24,000 |

The path to a living is **funded capital and patience**, not leverage. The engine is built;
this roadmap is how it earns.
