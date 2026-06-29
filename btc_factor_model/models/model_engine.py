"""
model_engine.py  -- PHASE 4: Elastic Net + XGBoost, walk-forward, attribution
=============================================================================

The forecasting/attribution layer. Given the causal z-scored feature panel from
Phase 3 (factor_engine) and the forward-return targets from Phase 1 (btc_returns),
it runs each model through the purged walk-forward and produces:

  1. fold_results -- per (model, horizon): the stitched out-of-sample prediction
     series, the matching per-date contribution frame, and aligned y_true.
  2. metrics      -- a tidy skill table: n, R², rank IC (Spearman), hit-rate,
     for every model × horizon, plus a rank-IC skill grid for the dashboard.
  3. attribution  -- per (model, horizon): feature contributions, the 11-parent
     category roll-up, today's decomposition, and a reconciliation residual.

Models, per the Phase-4 brief, are Elastic Net and XGBoost. Both expose the same
.fit/.predict/.contributions interface, and OLS / regime-switching can be added
to `models=` without any other change.

WHY THE ATTRIBUTION IS TRUSTWORTHY
----------------------------------
Elastic Net contributions are βᵢ·xᵢ on the standardized features; XGBoost uses
exact tree-SHAP from the booster. Both are additive and reconcile to the
prediction (Σ contributions + bias = prediction), so the category roll-up is a
faithful decomposition, not a heuristic. `reconciliation` reports the residual
(≈1e-15 for the linear model, ≈1e-7 for SHAP).

LOOK-AHEAD SAFETY (inherited, not re-implemented)
-------------------------------------------------
All of it lives in the walk-forward: origins march forward in time; training
rows whose h-day forward label overlaps the test window are purged (embargo = the
horizon); PCA is refit on each training fold only. This engine never fits on the
full sample — every prediction is genuinely out-of-sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ..config import SETTINGS, HORIZONS
from .estimators import (OLSModel, ElasticNetModel, XGBModel,
                         RegimeSwitchingModel)
from .walk_forward import WalkForward, FoldResult, evaluate
from ..attribution import contributions as attr

DEFAULT_MODELS = ("elastic_net", "xgboost")


@dataclass
class ModelResults:
    fold_results: dict = field(default_factory=dict)   # (model, horizon)->FoldResult
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    attribution: dict = field(default_factory=dict)    # (model, horizon)->tables
    skill_grid: pd.DataFrame = field(default_factory=pd.DataFrame)


class ModelEngine:
    """Run requested models through the walk-forward and attribute their output."""

    def __init__(self, models=DEFAULT_MODELS, settings=SETTINGS,
                 regime_signal: pd.Series | None = None):
        self.model_names = list(models)
        self.settings = settings
        self.model_cfg = settings.model
        self.rs = settings.random_state
        self.regime_signal = regime_signal
        self.wf = WalkForward(settings.wf, settings.pca)

    # ------------------------------------------------------------------ #
    # model factories (fresh estimator per fold)
    # ------------------------------------------------------------------ #
    def _factory(self, name: str):
        cfg, rs = self.model_cfg, self.rs
        if name == "elastic_net":
            return lambda: ElasticNetModel(cfg, rs)
        if name == "xgboost":
            return lambda: XGBModel(cfg, rs)
        if name == "ols":
            return lambda: OLSModel()
        if name == "regime_switching":
            if self.regime_signal is None:
                raise ValueError("regime_switching needs regime_signal=")
            return lambda: RegimeSwitchingModel(cfg, self.regime_signal, rs)
        raise ValueError(f"unknown model {name!r}")

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def run(self, Z: pd.DataFrame, targets: pd.DataFrame,
            pca_columns: list[str] | None = None,
            horizons: list[str] | None = None) -> ModelResults:
        horizons = horizons or list(HORIZONS.keys())
        res = ModelResults()
        rows = []

        for hz in horizons:
            for name in self.model_names:
                fr = self.wf.run(Z, targets, hz, self._factory(name),
                                 pca_columns=pca_columns)
                res.fold_results[(name, hz)] = fr

                m = evaluate(fr)
                rows.append({"model": name, "horizon": hz, **m})

                if len(fr.contrib):
                    res.attribution[(name, hz)] = {
                        "feature": attr.feature_contribution_table(fr.contrib),
                        "category": attr.category_contribution_table(fr.contrib),
                        "latest": attr.latest_decomposition(fr.contrib, fr.pred),
                        "reconciliation": attr.reconciliation_check(
                            fr.contrib, fr.pred),
                    }

        res.metrics = pd.DataFrame(rows)
        res.skill_grid = self._skill_grid(res.metrics, horizons)
        return res

    # ------------------------------------------------------------------ #
    def _skill_grid(self, metrics: pd.DataFrame,
                    horizons: list[str]) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame()
        grid = metrics.pivot(index="model", columns="horizon", values="rank_ic")
        return grid.reindex(columns=[h for h in horizons if h in grid.columns])


# --------------------------------------------------------------------------- #
# One-call end-to-end from an aligned panel (Phases 1+3+4 wired together)
# --------------------------------------------------------------------------- #
def run_model_engine(panel: pd.DataFrame, models=DEFAULT_MODELS,
                     settings=SETTINGS, horizons: list[str] | None = None,
                     regime_signal: pd.Series | None = None) -> ModelResults:
    """Build features with the Phase-3 FactorEngine, then run the models.

    `panel` must contain the dictionary variables, the five engineered originals,
    and the fwd_* target columns (from data/pipeline or btc_returns).
    """
    from ..features.factor_engine import FactorEngine

    eng = FactorEngine(settings)
    Z = eng.normalize(panel)
    eng.fit(Z)                                   # establishes the modeling universe
    universe = eng.universe_

    targets = panel[[h for h in HORIZONS if h in panel.columns]]
    me = ModelEngine(models, settings, regime_signal=regime_signal)
    return me.run(Z[universe["all"]], targets,
                  pca_columns=universe["pca_columns"], horizons=horizons)


# --------------------------------------------------------------------------- #
# QA helpers
# --------------------------------------------------------------------------- #
def skill_table(res: ModelResults) -> pd.DataFrame:
    """Tidy, rounded view of the skill metrics ordered by horizon."""
    order = {h: i for i, h in enumerate(HORIZONS)}
    m = res.metrics.copy()
    m["_o"] = m["horizon"].map(order)
    return (m.sort_values(["model", "_o"]).drop(columns="_o")
            .round({"r2": 3, "rank_ic": 3, "hit": 3})
            .reset_index(drop=True))


def reconciliation_summary(res: ModelResults) -> dict:
    return {f"{mdl}|{hz}": round(d["reconciliation"], 10)
            for (mdl, hz), d in res.attribution.items()}


if __name__ == "__main__":
    from ..data import synthetic, pipeline
    from ..factors.construct import add_original_factors

    syn = synthetic.generate()
    panel = pipeline.build(syn["factors"], syn["price"])  # adds fwd_* targets
    panel = add_original_factors(panel)
    panel = pipeline.add_forward_return_targets(panel)    # ensure targets present

    res = run_model_engine(panel, models=("elastic_net", "xgboost"))
    print("SKILL (rank IC rises with horizon; R² on levels stays hard):")
    print(skill_table(res).to_string(index=False))
    print("\nRANK-IC SKILL GRID:")
    print(res.skill_grid.round(3).to_string())
    print("\nRECONCILIATION (Σcontrib+bias vs prediction):")
    for k, v in reconciliation_summary(res).items():
        print(f"  {k:24s} {v:.2e}")
