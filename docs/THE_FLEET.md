# The Fleet — multi-edge research

> A complete, validated, risk-managed multi-strategy trading system, built and stress-tested
> across forex, equity indices, metals, and energy. Every edge earned its place through an
> out-of-sample robustness gauntlet; every dead end is documented too.

This is the research arc beyond the single-pair EURUSD model (see `README.md`). The goal:
turn one rare edge into a *fleet* of uncorrelated edges with enough frequency and low
enough drawdown to be **prop-firm fundable**.

---

## The method (why these numbers are trustworthy)

Every candidate strategy is a standalone harness measuring **net of realistic cost**, then
run through a **robustness gauntlet** before it's trusted:
1. **Out-of-sample** — split history in two halves; the edge must work in *both* (kills
   small-sample flukes and decayed edges).
2. **Parameter stability** — PF across a grid; a real edge is a *plateau*, an overfit one a
   lonely spike.
3. **Recent years** — is it still alive, or did it die after its golden era?

This gauntlet killed more ideas than it passed — which is exactly the point.

---

## Validated edges

**Reversion sleeve** (pays in chop / ranging):

| Edge | Market | Strategy | PF | Notes |
|------|--------|----------|----|-------|
| EURUSD | forex | VP mean-reversion (NEUTRAL regime) | 1.93 | the original specialist, rare (~1.6/yr) |
| S&P 500 | index | RSI(2) dip-buy in uptrend | 1.98 | failed momentum → excels at reversion |
| Nasdaq 100 | index | RSI(2) dip-buy in uptrend | 1.76 | trends *and* reverts |
| Dow 30 | index | RSI(2) dip-buy in uptrend | 1.36 | |

**Momentum sleeve** (pays in trends) — Donchian-480 breakout + ATR chandelier trail:

| Edge | Market | PF | Notes |
|------|--------|----|-------|
| DAX | index (long-only) | 1.42 | pure trender |
| Nikkei | index (long-only) | 1.41 | |
| Nasdaq 100 | index (long-only) | 1.32 | |
| Brent crude | energy | 1.21 | |
| Gold | metal | 1.18 | durable macro trend (real rates/USD) |
| Silver | metal | 1.17 | |

---

## Rejected by the gauntlet (the discipline that protects capital)

- **Multi-pair forex** — the VP reversion edge is EURUSD-specific; other majors trend.
- **Intraday VP** (EURUSD M5) — no edge (gross PF ~1.0; the market is too efficient).
- **Crypto VP reversion** — crypto trends, doesn't revert.
- **BTC/ETH momentum** — worked 2017–2021, *decayed* post-2021 (failed out-of-sample).
- **S&P / Dow / WTI / copper / nat-gas momentum** — failed OOS or param stability.
- **Gold/silver stat-arb** — the ratio *trends*, so it doesn't profitably mean-revert.

Lesson: real edges are rare *because* they aren't obvious — frequent, obvious edges on
liquid markets are arbitraged to zero.

---

## The combined portfolio (risk-managed)

All edges combined on one shared, **volatility-parity** equity curve (each edge scaled to
equal risk), with a risk layer: **per-sleeve budgeting** (correlated clusters share one
budget) + **de-risk-in-drawdown** + **vol-targeting**.

| Config | trades/yr | PF | CAGR | max DD | Sharpe |
|--------|-----------|----|------|--------|--------|
| flat (vol-parity) | 112 | 1.34 | 11.2% | −29.5% | 1.00 |
| + sleeve budgeting | 112 | 1.34 | 5.5% | **−15.4%** | 1.03 |
| + de-risk + vol-target 10% | 112 | 1.34 | 3.1% | **−9.5%** | 0.98 |

Sleeve budgeting **halved drawdown at the same Sharpe** — the 6 correlated momentum
instruments were the drawdown driver; treating them as one risk budget fixed it. The final
config is a smooth, fundable ~9.5%-drawdown curve over 21 years.

---

## The honest income path

**Income = Sharpe × funded capital.** At today's Sharpe ~1.0, a fundable 10% drawdown
buys ~3% CAGR ≈ **$5k/yr per $200k funded account** (80% profit split). Real, but not yet
a living on one account. Two levers, both clear:

1. **Raise the Sharpe** — add more *uncorrelated* edges. Sharpe 1.0 → ~3% CAGR at 10% DD;
   Sharpe 1.5 → ~7%; Sharpe 2.0 → ~10%+. Every edge lifts the ceiling.
2. **Scale the capital** — run the same system across multiple funded accounts.

The system is complete and fundable. Growing the income is now a matter of *more edges*
(higher Sharpe) and *more funded capital* (scale) — a known road, not a mystery.

---

## Tooling

| Script | Purpose |
|--------|---------|
| `scripts/download_dukascopy.py` | pull candles + real volume into DuckDB |
| `scripts/proto_momentum.py` | Donchian + ATR momentum (price-only, net-of-cost) |
| `scripts/proto_reversion_idx.py` | RSI(2) index mean-reversion |
| `scripts/proto_pairs.py` | market-neutral pairs/ratio reversion (stat-arb) |
| `scripts/robustness_momentum.py` | full gauntlet (OOS / param grid / year-by-year) |
| `scripts/screen_momentum.py`, `screen_reversion.py` | fleet screeners (rank many instruments) |
| `scripts/portfolio.py` | combine all edges → risk-managed equity curve |

Data (DuckDB) is gitignored — rebuild with `download_dukascopy.py`.
