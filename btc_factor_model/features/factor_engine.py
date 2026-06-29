"""
factor_engine.py  -- PHASE 3: normalization -> z-scores -> per-group PCA ->
                     parent factor scores
=====================================================================

This is the feature-construction brain. It takes the aligned raw panel (price +
dictionary variables + the five engineered originals) and produces three things:

  1. Z          -- the causal rolling z-scored panel (look-ahead-safe).
  2. features   -- the model-ready matrix: per-category PCA components + the five
                   engineered originals (kept standalone) + named passthroughs.
                   THIS is what the estimators in models/ consume.
  3. parent_scores -- ONE interpretable, bullish-oriented score per parent factor
                   category (11 columns). A summary/attribution view for EDA and
                   the dashboard; not the model input.

It composes the already-tested pieces rather than reinventing them:
  features/normalize.py  -> rolling_z (causal median/MAD, winsorized)
  features/plan.py       -> what gets PCA'd vs kept standalone vs passed through
  features/pca.py        -> CategoryPCA (per-group PCA, bullish sign-alignment)

CAUSALITY / LOOK-AHEAD
----------------------
* Z-scores use trailing windows only, so they are real-time computable.
* PCA is the one fitted step. The class is sklearn-style: in the walk-forward
  harness you `fit(Z_train)` then `transform(Z_test)` — never the reverse. Used
  full-sample via `fit_transform`, the parent scores are *descriptive* (in-sample
  loadings) and labelled as such; the forecasting path always refits per fold.

PARENT FACTOR SCORE -- definition
---------------------------------
For each parent category the score is the bullish-oriented average of:
  * its first principal component (already sign-aligned bullish by CategoryPCA), and
  * any engineered-standalone / passthrough members of that category, each
    oriented by its economic sign so that +1 sigma always means 'tailwind'.
The result is re-standardized (rolling z) so all 11 scores are directly
comparable on a common sigma scale.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import SETTINGS, PARENT_CATEGORIES
from .. import data_dictionary as dd
from .normalize import rolling_z, expanding_z, normalize_frame
from .plan import modeling_universe, ENGINEERED_STANDALONE, PASSTHROUGH_RAW
from .pca import CategoryPCA
from ..factors.transforms import apply_transform

# --------------------------------------------------------------------------- #
# Orientation of the engineered standalones: +1 if a HIGH value is bullish for
# forward BTC returns, -1 if bearish. (Raw passthroughs use the dictionary's
# expected sign.) These are the economic priors, identical to those used for
# PCA sign-alignment, applied here so every parent score reads "+ = tailwind".
# --------------------------------------------------------------------------- #
# Slow structural factors: normalize with an EXPANDING window so they stay
# mean-reverting (like the cone) instead of being turned into momentum by a
# 1-year rolling z-score.
SLOW_EXPANDING = {"power_law_dev", "mayer_multiple", "mvrv_z", "nupl"}

_STANDALONE_BULLISH = {
    "effective_float": -1,             # rising tradable float = more sellable supply
    "deriv_notional_over_float": -1,   # leverage stacked on thin float = fragility
    "treasury_company_factor": +1,     # corporate accumulation = demand
    "power_law_dev": -1,               # stretched above trend = headwind
    "funding_stress_factor": -1,       # carry-unwind stress = headwind
}
_STANDALONE_CATEGORY = {
    "effective_float": "effective_float",
    "deriv_notional_over_float": "derivatives",
    "treasury_company_factor": "treasury_demand",
    "power_law_dev": "valuation",
    "funding_stress_factor": "funding_stress",
}


@dataclass
class FactorScores:
    Z: pd.DataFrame                 # causal z-scored raw panel
    features: pd.DataFrame          # model-ready matrix (PCs + standalone + passthru)
    parent_scores: pd.DataFrame     # 11 bullish-oriented parent factor scores
    universe: dict                  # plan split (pca/standalone/passthrough/all)
    loadings: dict                  # per-category PCA loading matrices


class FactorEngine:
    """Normalize -> per-group PCA -> parent scores, as one fit/transform object."""

    def __init__(self, settings=SETTINGS):
        self.norm_cfg = settings.norm
        self.pca_cfg = settings.pca
        self.pca = CategoryPCA(self.pca_cfg)
        self.universe_: dict | None = None
        self.feature_cols_: list[str] | None = None
        self._cat_of = {v.id: v.category for v in dd.VARIABLES}
        self._exp_sign = dd.expected_signs()
        self._transform_of = {v.id: v.transform for v in dd.VARIABLES}

    # ------------------------------------------------------------------ #
    # column selection + normalization
    # ------------------------------------------------------------------ #
    def feature_columns(self, panel: pd.DataFrame) -> list[str]:
        """Dictionary variables + engineered originals present in the panel.
        Excludes price/OHLCV/return targets."""
        dict_ids = [v.id for v in dd.VARIABLES if v.id in panel.columns]
        extra = [c for c in ENGINEERED_STANDALONE if c in panel.columns
                 and c not in dict_ids]
        return dict_ids + extra

    def normalize(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply each variable's dictionary transform (yoy / log-diff / diff /
        level) to make trending series stationary, THEN causal rolling z-score.
        The transform field used to be ignored, which let non-stationary levels
        (M2, supply, addresses, net liquidity) leak a trend into the models and
        fool the linear estimator. Engineered originals have no dict transform
        and pass through as 'level' (they are already stationary composites)."""
        self.feature_cols_ = self.feature_columns(panel)
        tf = pd.DataFrame(index=panel.index)
        for c in self.feature_cols_:
            tf[c] = apply_transform(panel[c], self._transform_of.get(c, "level"))
        tf = tf.replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame(index=panel.index)
        for c in self.feature_cols_:
            zfn = expanding_z if c in SLOW_EXPANDING else rolling_z
            out[c] = zfn(tf[c], self.norm_cfg)
        return out

    # ------------------------------------------------------------------ #
    # fit / transform (PCA is the only fitted step)
    # ------------------------------------------------------------------ #
    def fit(self, Z_train: pd.DataFrame) -> "FactorEngine":
        self.universe_ = modeling_universe(list(Z_train.columns))
        self.pca.fit(Z_train[self.universe_["all"]],
                     pca_columns=self.universe_["pca_columns"])
        return self

    def transform(self, Z: pd.DataFrame) -> pd.DataFrame:
        if self.universe_ is None:
            raise RuntimeError("call fit() before transform().")
        return self.pca.transform(Z[self.universe_["all"]])

    # ------------------------------------------------------------------ #
    # parent factor scores (the Phase-3 headline output)
    # ------------------------------------------------------------------ #
    def parent_scores(self, features: pd.DataFrame,
                      restandardize: bool = True) -> pd.DataFrame:
        """One bullish-oriented score per parent category from the transformed
        features. PC1 (already bullish) + oriented standalone/passthrough members."""
        out = {}
        for cat in PARENT_CATEGORIES:
            members = []

            pc1 = f"{cat}_pc1"
            if pc1 in features.columns:
                members.append(features[pc1])              # already bullish

            for s, scat in _STANDALONE_CATEGORY.items():
                if scat == cat and s in features.columns:
                    members.append(_STANDALONE_BULLISH[s] * features[s])

            for p in PASSTHROUGH_RAW:
                if p in features.columns and self._cat_of.get(p) == cat:
                    sign = self._exp_sign.get(p, 0) or 1   # 0 -> treat as +1
                    members.append(sign * features[p])

            if members:
                score = pd.concat(members, axis=1).mean(axis=1)
                out[cat] = rolling_z(score, self.norm_cfg) if restandardize else score

        return pd.DataFrame(out, index=features.index)

    # ------------------------------------------------------------------ #
    # convenience: full-sample descriptive run
    # ------------------------------------------------------------------ #
    def fit_transform(self, panel: pd.DataFrame) -> FactorScores:
        """Descriptive (in-sample) run for EDA / dashboard. The predictive path
        refits per fold inside the walk-forward instead."""
        Z = self.normalize(panel)
        self.fit(Z)
        features = self.transform(Z)
        parents = self.parent_scores(features)
        return FactorScores(Z=Z, features=features, parent_scores=parents,
                            universe=self.universe_, loadings=self.pca.loadings())

    # ------------------------------------------------------------------ #
    # interpretability
    # ------------------------------------------------------------------ #
    def pca_summary(self) -> pd.DataFrame:
        """Per-category: #components kept and cumulative variance explained."""
        rows = []
        for cat, model in self.pca.models.items():
            evr = model.explained_variance_ratio_
            rows.append({"category": cat, "n_inputs": len(self.pca.cat_cols[cat]),
                         "n_components": len(evr),
                         "cum_var_explained": round(float(evr.sum()), 3)})
        for cat, cols in self.pca.passthrough.items():
            rows.append({"category": cat, "n_inputs": len(cols),
                         "n_components": 0, "cum_var_explained": np.nan})
        return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)


def run_factor_engine(panel: pd.DataFrame, settings=SETTINGS) -> FactorScores:
    """One-call descriptive factor build from an aligned panel."""
    return FactorEngine(settings).fit_transform(panel)


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
def qa_report(fs: FactorScores) -> dict:
    feats = fs.features
    parents = fs.parent_scores
    # winsor bound check on z-scores
    zmax = float(np.nanmax(np.abs(fs.Z.values)))
    return {
        "z_columns": fs.Z.shape[1],
        "n_model_features": feats.shape[1],
        "feature_names": list(feats.columns),
        "parent_categories": list(parents.columns),
        "z_within_winsor": bool(zmax <= SETTINGS.norm.winsor_z + 1e-9),
        "parent_scores_centered": {
            c: round(float(parents[c].dropna().mean()), 3) for c in parents.columns
        },
        "pca_categories": list(fs.loadings.keys()),
    }


if __name__ == "__main__":
    # lightweight offline demo: build a panel, run the engine.
    from ..data import synthetic
    from ..data import pipeline
    from ..factors.construct import add_original_factors

    syn = synthetic.generate()
    panel = pipeline.align_panel(syn["factors"], syn["price"])
    panel = add_original_factors(panel)

    eng = FactorEngine()
    fs = eng.fit_transform(panel)
    print("model features:", fs.features.shape[1])
    print(fs.features.columns.tolist())
    print("\nPCA by factor group:")
    print(eng.pca_summary().to_string(index=False))
    print("\nparent factor scores (latest):")
    print(fs.parent_scores.dropna().iloc[-1].round(2).sort_values().to_string())
