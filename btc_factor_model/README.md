# BTC Factor Model

An institutional-style framework for **explaining and attributing** Bitcoin returns across six
horizons (1d, 5d, 20d, 60d, 6m, 12m) using eleven macro / crypto factor families.

> **Mandate: explain first, forecast second.** The system is built as an *attribution engine*.
> Point-forecast accuracy on return *levels* is poor at every horizon (as it should be); the
> usable signal is *cross-sectional / rank* skill that grows with horizon and is strongest at
> 60d–12m. The outputs make that honesty explicit.

---

## ⚠️ Data status

This sandbox cannot reach FRED / Coinglass / Glassnode / Farside / exchange APIs, so the package
ships two interchangeable data layers behind one schema:

- **`data/sources.py`** — real-source *adapter stubs*. Each raises `NotImplementedError` with the
  exact series IDs / endpoints to wire up (FRED `M2SL`, `WALCL`, `WTREGEN`, `RRPONTSYD`, `DTWEXBGS`,
  `DFII10`, `VIXCLS`, `BAMLH0A0HYM2`, `DGS2`; Deribit/Coinglass derivatives; Glassnode on-chain;
  Farside ETF flows from 2024-01-11; bitcointreasuries corporate holdings). Drop in your keys.
- **`data/synthetic.py`** — a clearly-labelled **synthetic** generator that mirrors the real schema
  (power-law price + AR(1) latent macro drivers + fat-tailed noise) so the whole pipeline runs
  offline. **It is not real data and must not be used for any live decision.**

Everything downstream (lags, normalization, PCA, models, walk-forward, attribution, dashboard) is
identical for both layers. Switch with `run(source="real")` once adapters are implemented.

---

## The five original factors (`factors/construct.py`)

All are computed causally (no look-ahead — expanding windows / running sums only):

1. **Effective Tradable Float** — circulating supply minus illiquid/lost coins, structural
   sinks, and the non-overlapping portion of treasury + ETF holdings (0.5 overlap guard against
   double-counting custody).
2. **Derivative Notional / Effective Float** — (futures OI + options OI) in USD divided by the
   *dollar value of the effective float*. A leverage-vs-real-supply stress gauge.
3. **Treasury Company Factor** — blended causal z-score of corporate net BTC buys (scaled by
   supply) and treasury-vehicle mNAV premium.
4. **Bitcoin Power-Law Deviation** — log price minus a power-law fair value fit by *expanding* OLS
   on log(days-since-genesis); deviation is the rich/cheap signal.
5. **Funding Stress Factor** — USD/JPY strength + JPY-vol z-scores (+ carry compression). A carry-
   unwind / global-funding proxy.

---

## Look-ahead safeguards (three independent layers)

1. **Publication lags** — every variable carries `pub_lag_days` in the data dictionary and is
   shifted forward by that many days before it can be used (`pipeline.apply_publication_lags`).
2. **Causal feature engineering** — rolling z-scores, PCA sign-alignment, power-law fit, and
   regime labels all use expanding/trailing windows only.
3. **Purged, embargoed walk-forward** — `models/walk_forward.py` purges the training tail by the
   horizon length (embargo) so overlapping forward-return labels can't leak into training. PCA is
   refit per fold on the training slice only. **Walk-forward is the only validation used.**

---

## Pipeline

```
data → align_panel → apply_publication_lags → add_original_factors
     → forward-return targets → rolling z-score → per-category PCA (train-only)
     → {OLS, ElasticNet, XGBoost, RegimeSwitching} × 6 horizons
     → purged walk-forward → contribution tables → dashboard
```

PCA is applied *within* parent categories (keeps components to 90% var, max 3), and the five
engineered factors are kept standalone (never folded into PCA) so attribution stays interpretable.
Contributions are exact: linear models via β·x, XGBoost via tree-SHAP — both reconcile to the
model prediction (≈1e-7 or better).

## Run it

```bash
cd ..                      # parent of the package dir
python -c "from btc_factor_model import run, dashboard; \
           res = run.run(source='synthetic'); \
           dashboard.build_dashboard(res, out_path='btc_factor_model/artifacts/dashboard.html')"
```

## Artifacts (`artifacts/`)

| file | what |
|---|---|
| `dashboard.html` | self-contained daily dashboard (open in a browser) |
| `data_dictionary.csv` / `.md` | all 39 variables × 11 categories, with sources, lags, expected signs |
| `skill_metrics.csv` | rank-IC / R² / hit-rate / n for every model × horizon |
| `category_contributions_60d_xgb.csv` | parent-category attribution (headline 60d view) |
| `feature_contributions_60d_xgb.csv` | per-feature attribution |
| `latest_decomposition_60d_xgb.csv` | today's drivers, ranked |

## Honest read on the synthetic results

Rank IC rises with horizon (~0.09 at 1d → ~0.32 at 60d → ~0.34 at 6m) then fades at 12m as sample
shrinks; OOS R² on return *levels* is negative at long horizons for linear/regime models and least-bad
for XGBoost — i.e. the framework *ranks* regimes far better than it *forecasts* magnitudes. That is the
intended, defensible result for a macro attribution model. On real data the numbers will differ; the
machinery and the safeguards are what carry over.

---

*Research tooling for factor attribution. Not investment advice.*
