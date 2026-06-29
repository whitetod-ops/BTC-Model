# CLAUDE.md — BTC Factor Model

> This file is the project's source of truth. Claude Code loads it automatically.
> It holds the goal, the architecture, how to run things, how to refine the model,
> and the conventions that must not be broken (especially look-ahead safety).

---

## 1. What this project is

An institutional-style framework that **explains and attributes** Bitcoin returns
across six horizons (1d, 5d, 20d, 60d, 6m, 12m) using eleven macro / crypto factor
families, and projects an **upside / fair / downside valuation cone** from 30 days
to 3 years.

**Mandate: explain first, forecast second.** This is an *attribution + valuation*
engine, not a trading signal. Point-forecast accuracy on return *levels* is poor at
every horizon (correctly — see the skill table).

REALITY CHECK (measured on REAL data via backtest.py, not assumed): the ML factor
grid has **no reliable out-of-sample edge** — purged-walk-forward rank IC at 3-6m is
near zero (ElasticNet often negative), and overlap-adjusted p-values are not
significant (~6-27 independent windows is the data ceiling). The earlier "rank IC
~0.35 at 6m" was SYNTHETIC-only and did NOT hold on real data; treat it as a
disproven hypothesis, not a result. The genuinely validated signal is the
**valuation cone**: ~90% channel calibration and a cheap>fair>rich forward-return
ordering that survived a true frozen holdout (fit pre-2018, tested on unseen 2018+).
Use this model for *where BTC sits vs fair value*, not for directional forecasts.

The owner is a quantitative macro researcher who wants, for *today*: where BTC sits
versus fair value, which factors are tailwinds vs headwinds, and the plausible range
from here over 30 days to 2+ years. Not an hourly trader.

---

## 2. Current status

- **Runs end to end on synthetic data today**, no API keys required:
  `python -m btc_factor_model.daily_run`
- The whole pipeline (panel → factors → PCA → walk-forward models → attribution →
  dashboard → valuation cone) is built and tested.
- **What's left to go live:** wire the real data adapters in `data/sources.py`
  (see `SETUP_DATA_FEEDS.md`). Everything downstream is identical for synthetic vs
  real — the synthetic generator mirrors the real schema exactly.
- Numbers produced right now are from **synthetic data** and are for plumbing only.

---

## 3. The five original factors

All computed causally (expanding windows / running sums — no look-ahead):

1. **Effective Tradable Float** — circulating supply minus illiquid/lost coins,
   structural sinks, and the non-overlapping part of treasury + ETF holdings
   (0.5 overlap guard against double-counting custody).
2. **Derivative Notional / Effective Float** — (futures OI + options OI) in USD over
   the dollar value of the float. Leverage-vs-real-supply fragility gauge.
3. **Treasury Company Factor** — causal z-blend of corporate net BTC buys (scaled by
   supply) and treasury-vehicle mNAV premium.
4. **Bitcoin Power-Law Deviation** — log price minus an expanding-OLS power-law fair
   value; the rich/cheap signal. Also the backbone of the valuation cone.
5. **Funding Stress Factor** — USD/JPY strength + JPY-vol z-scores (+ carry
   compression). Carry-unwind / global-funding proxy.

---

## 4. Architecture (file map)

```
btc_factor_model/
├── CLAUDE.md                  ← you are here
├── SETUP_DATA_FEEDS.md        ← how to get the APIs / scrapers (read this to go live)
├── requirements.txt
├── daily_run.py               ← ONE command runs everything (synthetic or real)
│
├── config.py                  horizons, 11 categories, all hyperparameters
├── data_dictionary.py         single source of truth: 39 vars, sources, pub lags, expected signs
│
├── data/
│   ├── sources.py             REAL adapter stubs — fill these in to go live
│   ├── synthetic.py           offline generator (mirrors the real schema; NOT real data)
│   ├── btc_returns.py         Phase 1: BTC OHLCV + forward-return targets
│   ├── macro_data.py          Phase 2: FRED/Yahoo macro block (DXY, VIX, MOVE, yields, net liquidity…)
│   └── pipeline.py            align panel · apply publication lags · build forward-return targets
│
├── factors/
│   ├── transforms.py          causal transforms (log_diff, yoy, decay…)
│   └── construct.py           the five original engineered factors
│
├── features/
│   ├── normalize.py           causal rolling z-scores (robust median/MAD, winsorized)
│   ├── plan.py                what gets PCA'd vs kept standalone vs passthrough
│   ├── pca.py                 per-category PCA with bullish sign-alignment
│   └── factor_engine.py       Phase 3: normalize → PCA → 11 parent factor scores
│
├── models/
│   ├── estimators.py          OLS / ElasticNet / XGBoost / RegimeSwitching (shared interface)
│   ├── walk_forward.py        purged + embargoed walk-forward (the ONLY validation)
│   └── model_engine.py        Phase 4: run models → skill grid + reconciling attribution
│
├── attribution/
│   └── contributions.py       feature & 11-parent-category contribution tables
│
├── valuation_cone.py          power-law channel + scenario bands + regime tilt (upside/fair/downside)
├── dashboard_daily.py         Phase 5: today's read (scores, regime, prediction, attribution, residual)
├── dashboard.py               original historical valuation/skill dashboard
└── run.py                     original end-to-end driver (4 models × 6 horizons)
```

---

## 5. How to run it

```bash
# from the PARENT directory of btc_factor_model/
python -m btc_factor_model.daily_run                       # synthetic, EN + XGB
python -m btc_factor_model.daily_run --models elastic_net xgboost ols regime_switching
python -m btc_factor_model.daily_run --horizon fwd_60d
python -m btc_factor_model.daily_run --source real         # after wiring data/sources.py
```

Writes to `artifacts/`: `daily_dashboard.html`, `valuation_cone.html`,
`skill_metrics.csv`, `parent_scores.csv`. Open the two HTML files in a browser.

Individual phases are runnable too (each module has a `__main__` demo), e.g.:
```bash
python -m btc_factor_model.models.model_engine
python -m btc_factor_model.valuation_cone
```

---

## 6. Look-ahead safety — DO NOT BREAK THIS

Three independent layers keep the model honest. Any change must preserve all three:

1. **Publication lags.** Every variable carries `pub_lag_days` in `data_dictionary.py`
   and is shifted forward by that many days in `pipeline.apply_publication_lags`
   before it can be used. (Data adapters return data stamped at its *observation*
   date; lagging happens here, once — never double-lag in an adapter.)
2. **Causal feature engineering.** Rolling z-scores, PCA sign-alignment, the
   power-law fit, and regime labels all use expanding/trailing windows only.
3. **Purged, embargoed walk-forward** (`models/walk_forward.py`). Training rows whose
   h-day forward label overlaps the test window are purged; the embargo equals the
   horizon. PCA is refit on each training fold only. This is the **only** validation —
   no random splits, no full-sample fits in the forecast path.

Attribution is exact and reconciles: linear via β·x, XGBoost via tree-SHAP, both
summing to the prediction (~1e-7 residual). The dashboard's RESIDUAL panel is that check.

---

## 7. Conventions (read before editing)

- **The data contract is everything.** Every data adapter returns a daily DataFrame
  indexed by date whose **column names are dictionary IDs** (`data_dictionary.py`).
  Nothing downstream cares where the data came from. Add a new variable = add a
  `Variable(...)` row to the dictionary (with its `pub_lag_days` and `exp_sign`) and
  return that column from an adapter.
- **Synthetic mirrors real.** `data/synthetic.py` produces the same columns as the
  real adapters so the pipeline runs offline. Keep them in sync when adding variables.
- **Engineered factors stay standalone.** The five originals are never folded into
  PCA (`features/plan.py`), so they keep a named line in every attribution table.
- **Causality first.** If you add a feature, it must be computable in real time
  (trailing windows only). If you're unsure, it's probably look-ahead.
- **Honesty in outputs.** Report rank IC *and* R². Negative long-horizon R² is the
  correct result, not a bug to tune away.
- **Not investment advice.** This is research tooling. Outputs say so; keep it that way.

---

## 8. How to refine the model (the simulation loop)

The whole point of moving to Claude Code: run simulations on real (or synthetic) data
and iterate. The loop:

1. **Get a baseline.** Run `daily_run` on synthetic, note the `skill_metrics.csv`
   (rank IC by model × horizon). That's your reference.
2. **Wire real data** incrementally (see `SETUP_DATA_FEEDS.md`): FRED + Yahoo first
   (≈60% of the model — all macro + the power-law backbone), then Coin Metrics
   (on-chain), then the Farside ETF scraper, then derivatives. Validate each addition
   against the synthetic baseline before adding the next.
3. **Re-run and read the skill grid.** Rank IC should rise with horizon. If a factor
   adds nothing, the attribution table will show ~0 contribution — consider dropping
   or reworking it.
4. **Tune deliberately, validate honestly.** Hyperparameters live in `config.py`
   (z-score window, PCA variance target, XGBoost depth, walk-forward train/test/step).
   Change them, re-run the walk-forward, compare OOS rank IC. Never tune on the test
   blocks; the purged walk-forward is the judge.
5. **Add / revise factors** in `factors/construct.py` (+ dictionary row + synthetic
   column). Re-run, check the new factor's attribution and whether OOS skill improved.
6. **Read the cone.** `valuation_cone.py` turns the factor regime into an upside/fair/
   downside envelope. Feed it real price history + the parent scores.

Cadence for the owner's use case (30d–2yr valuation, not trading): **infer weekly**
(after the Thursday Fed balance-sheet release), **refit/revalidate monthly or
quarterly**. Daily is overkill — long-horizon projections barely move day to day.

Good tasks to hand Claude Code:
- "Implement `macro_liquidity_block` in data/sources.py using the FRED snippets in
  SETUP_DATA_FEEDS.md, then run daily_run --source real and show me the skill grid."
- "Add a 2-year and 3-year forward-return target and re-validate; flag if there's
  enough data to trust them."
- "Drop the valuation cone in as a panel in dashboard_daily.py."
- "The regulatory factor contributes ~0 — investigate and either fix or remove it."

---

## 9. Conversation summary (how this was built)

This project was built in phases, each a tested module, with a consistent
"explain-first / don't-overclaim" philosophy:

- **Foundation:** data dictionary (39 variables × 11 categories, each with a
  publication lag and an expected sign), config, and the real-vs-synthetic data
  contract. The sandbox it was built in had no external network, so a synthetic
  generator that mirrors the real schema was built first so everything runs offline.
- **Phase 1 — `btc_returns.py`:** BTC OHLCV ingestion (yfinance / CSV / synthetic)
  + look-ahead-safe forward-return targets, with a QA check proving the targets'
  NaN tail points forward.
- **Phase 2 — `macro_data.py`:** the FRED/Yahoo macro block. Handles the real-world
  unit gotcha (Fed balance sheet & TGA in millions, RRP in billions) when building
  net liquidity; does NOT apply publication lags (that's the pipeline's job).
- **Phase 3 — `factor_engine.py`:** normalize → per-category PCA → 11 bullish-oriented
  parent factor scores, as one fit/transform object (so it's causal inside the
  walk-forward).
- **Phase 4 — `model_engine.py`:** Elastic Net + XGBoost through the purged
  walk-forward, producing a rank-IC skill grid and reconciling attribution. (Fixed an
  sklearn `n_alphas` deprecation along the way.)
- **Phase 5 — `dashboard_daily.py`:** the daily cockpit — factor scores, regime,
  predicted return, attribution, and the reconciliation residual.
- **Valuation cone — `valuation_cone.py`:** power-law channel (structural upside/fair/
  downside) + empirical return-dispersion band (realistic near-term width) + a bounded,
  horizon-fading regime tilt that locates the most-likely path inside the envelope.
- **Then:** a Q&A on free APIs to use, where to run it (Claude Code, locally — the
  build sandbox couldn't reach data vendors), and run cadence (weekly infer / monthly
  refit), plus a step-by-step data-feed setup runbook (now `SETUP_DATA_FEEDS.md`).

Validated behavior. On SYNTHETIC data the plumbing checks out: attribution
reconciles to ~1e-7, all five originals appear by name, the cone's regime tilt is
monotone and channel-bounded. On REAL data (the honest test): the factor-model rank
IC is weak/insignificant at every horizon — do not trust it for forecasting — while
the valuation cone is well-calibrated and held a frozen out-of-sample holdout. A
second fair-value anchor (classic Metcalfe, value~active-addr^2) is reported
alongside the power-law as a cross-check; the channel itself stays pure power-law.
Diagnostics live in backtest.py: --cone, --cone-holdout, --anchors, --regime.

---

## 10. Honest caveats (keep these visible)

- Long-horizon return *levels* are not forecastable from ~8 years of data; rank IC and
  the valuation cone are the trustworthy outputs, not 2-year point predictions.
- The power law is an empirical regularity, not a law — the cone assumes it keeps
  holding, and BTC's slope may be flattening as the asset matures.
- Glassnode's proprietary on-chain cohorts (illiquid/LTH supply) that feed Effective
  Float are paid; free substitutes (Coin Metrics) are coarser. De-weight or proxy
  until you decide it's worth paying for.
- Research tooling. Not investment advice.
