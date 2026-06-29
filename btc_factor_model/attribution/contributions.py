"""
Attribution engine (step 9): factor contribution tables.

Takes the out-of-sample per-date contribution frame from any model (linear betas
or XGBoost SHAP -- both additive and prediction-reconciling) and turns it into
the tables an investment committee actually reads:

  * average contribution per feature (which factors explain returns)
  * contribution rolled up to the 11 parent categories
  * a point-in-time decomposition for the latest date (today's drivers)
  * share of explained variation per factor

Because contributions sum to the prediction (plus bias), every table reconciles:
sum of factor contributions + bias = model prediction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import PARENT_CATEGORIES


def _feature_to_category(feature: str) -> str:
    """Map a model feature back to a parent category for roll-ups."""
    # category PCs are named '<category>_pc<k>'
    for cat in PARENT_CATEGORIES:
        if feature.startswith(cat + "_pc"):
            return cat
    # engineered / passthrough standalone features
    explicit = {
        "effective_float": "effective_float",
        "deriv_notional_over_float": "derivatives",
        "treasury_company_factor": "treasury_demand",
        "power_law_dev": "valuation",
        "funding_stress_factor": "funding_stress",
        "etf_net_flow": "etf_flows",
        "etf_flow_5d": "etf_flows",
        "reg_event_score": "regulatory",
    }
    return explicit.get(feature, "other")


def feature_contribution_table(contrib: pd.DataFrame) -> pd.DataFrame:
    """Average absolute & signed contribution per feature across the OOS sample."""
    feats = [c for c in contrib.columns if c != "bias"]
    c = contrib[feats]
    tab = pd.DataFrame({
        "mean_contrib": c.mean(),
        "mean_abs_contrib": c.abs().mean(),
        "std_contrib": c.std(),
    })
    tab["share_of_abs"] = tab["mean_abs_contrib"] / tab["mean_abs_contrib"].sum()
    tab["category"] = [_feature_to_category(f) for f in tab.index]
    return tab.sort_values("mean_abs_contrib", ascending=False)


def category_contribution_table(contrib: pd.DataFrame) -> pd.DataFrame:
    """Roll feature contributions up to the 11 parent categories."""
    feats = [c for c in contrib.columns if c != "bias"]
    cats = pd.Series({f: _feature_to_category(f) for f in feats})
    by_cat_signed = contrib[feats].T.groupby(cats).sum().T
    tab = pd.DataFrame({
        "mean_contrib": by_cat_signed.mean(),
        "mean_abs_contrib": by_cat_signed.abs().mean(),
    })
    tab["share_of_abs"] = tab["mean_abs_contrib"] / tab["mean_abs_contrib"].sum()
    return tab.sort_values("mean_abs_contrib", ascending=False)


def latest_decomposition(contrib: pd.DataFrame, pred: pd.Series,
                         top_n: int = 12) -> pd.DataFrame:
    """Today's prediction decomposed into factor pushes (the dashboard's core)."""
    last = contrib.iloc[-1]
    feats = [c for c in contrib.columns if c != "bias"]
    d = (last[feats].rename("contribution").to_frame())
    d["category"] = [_feature_to_category(f) for f in d.index]
    d["abs"] = d["contribution"].abs()
    d = d.sort_values("abs", ascending=False).drop(columns="abs")
    d.attrs["bias"] = float(last.get("bias", 0.0))
    d.attrs["prediction"] = float(pred.iloc[-1]) if len(pred) else np.nan
    d.attrs["date"] = contrib.index[-1]
    return d.head(top_n)


def reconciliation_check(contrib: pd.DataFrame, pred: pd.Series) -> float:
    """Max absolute gap between summed contributions and the prediction.
    Should be ~0 for additive models (sanity check on the attribution)."""
    summed = contrib.sum(axis=1)
    common = summed.index.intersection(pred.index)
    return float((summed.reindex(common) - pred.reindex(common)).abs().max())
