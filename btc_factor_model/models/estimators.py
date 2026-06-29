"""
Estimators (step 6) with a single shared interface so the walk-forward harness
and the attribution engine can treat them interchangeably:

    .fit(X, y) -> self
    .predict(X) -> np.ndarray
    .contributions(X) -> DataFrame[feature contributions] + 'bias' column

Contribution semantics:
  * Linear (OLS, ElasticNet): contribution_i = beta_i * x_i, bias = intercept.
    These sum exactly to the prediction -- clean additive attribution.
  * XGBoost: exact tree-SHAP via the booster's built-in pred_contribs (no
    external shap dependency). SHAP values also sum to the prediction.
  * Regime-switching: probability-weighted blend of per-regime linear models;
    contributions are the prob-weighted blend of each regime's beta_i * x_i.

All four therefore yield additive, prediction-reconciling attributions, which is
exactly what an 'explain first' mandate needs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from ..config import ModelConfig


# --------------------------------------------------------------------------- #
# Linear models
# --------------------------------------------------------------------------- #
class OLSModel:
    name = "ols"

    def fit(self, X, y):
        self.cols_ = list(X.columns)
        self.m_ = LinearRegression().fit(X.values, y.values)
        return self

    def predict(self, X):
        return self.m_.predict(X[self.cols_].values)

    def contributions(self, X):
        c = X[self.cols_].values * self.m_.coef_
        df = pd.DataFrame(c, index=X.index, columns=self.cols_)
        df["bias"] = self.m_.intercept_
        return df


class ElasticNetModel:
    name = "elastic_net"

    def __init__(self, cfg: ModelConfig, random_state=7):
        self.cfg = cfg
        self.rs = random_state

    def fit(self, X, y):
        self.cols_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.values)
        Xs = self.scaler_.transform(X.values)
        common = dict(l1_ratio=list(self.cfg.elastic_net_l1_ratios),
                      cv=5, random_state=self.rs, max_iter=20000)
        # sklearn >=1.7 takes an int via `alphas`; older versions use `n_alphas`.
        try:
            self.m_ = ElasticNetCV(alphas=self.cfg.elastic_net_alphas,
                                   **common).fit(Xs, y.values)
        except TypeError:
            self.m_ = ElasticNetCV(n_alphas=self.cfg.elastic_net_alphas,
                                   **common).fit(Xs, y.values)
        return self

    def predict(self, X):
        return self.m_.predict(self.scaler_.transform(X[self.cols_].values))

    def contributions(self, X):
        Xs = self.scaler_.transform(X[self.cols_].values)
        c = Xs * self.m_.coef_
        df = pd.DataFrame(c, index=X.index, columns=self.cols_)
        df["bias"] = self.m_.intercept_
        return df


# --------------------------------------------------------------------------- #
# XGBoost with exact tree-SHAP contributions
# --------------------------------------------------------------------------- #
class XGBModel:
    name = "xgboost"

    def __init__(self, cfg: ModelConfig, random_state=7):
        self.params = dict(cfg.xgb_params)
        self.params["seed"] = random_state
        self.n_estimators = self.params.pop("n_estimators", 300)

    def fit(self, X, y):
        self.cols_ = list(X.columns)
        dtrain = xgb.DMatrix(X.values, label=y.values, feature_names=self.cols_)
        self.booster_ = xgb.train(self.params, dtrain,
                                  num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        d = xgb.DMatrix(X[self.cols_].values, feature_names=self.cols_)
        return self.booster_.predict(d)

    def contributions(self, X):
        d = xgb.DMatrix(X[self.cols_].values, feature_names=self.cols_)
        shap = self.booster_.predict(d, pred_contribs=True)  # (n, k+1), last=bias
        df = pd.DataFrame(shap[:, :-1], index=X.index, columns=self.cols_)
        df["bias"] = shap[:, -1]
        return df


# --------------------------------------------------------------------------- #
# Regime-switching: Markov regimes on returns + per-regime ridge, prob-weighted
# --------------------------------------------------------------------------- #
class RegimeSwitchingModel:
    name = "regime_switching"

    def __init__(self, cfg: ModelConfig, regime_signal: pd.Series,
                 random_state=7):
        self.cfg = cfg
        self.regime_signal = regime_signal   # causal series (e.g. trailing vol)
        self.rs = random_state

    def _label_regimes(self, index):
        """Causal 2-state labels from the regime signal via an expanding median
        threshold (robust; avoids per-fold Markov convergence headaches).
        State 1 = high-stress/high-vol, State 0 = calm."""
        s = self.regime_signal.reindex(index)
        thresh = s.expanding(min_periods=60).median()
        return (s > thresh).astype(int).fillna(0)

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        self.cols_ = list(X.columns)
        lab = self._label_regimes(X.index)
        self.models_ = {}
        for r in (0, 1):
            mask = (lab == r).values
            if mask.sum() < max(40, X.shape[1] + 5):
                mask = np.ones(len(X), bool)        # fallback: pool
            self.models_[r] = Ridge(alpha=1.0).fit(
                X.values[mask], y.values[mask])
        # base-rate regime prob from training (used if signal missing at test)
        self.base_p1_ = float(lab.mean())
        return self

    def _regime_prob(self, index):
        s = self.regime_signal.reindex(index)
        thresh = s.expanding(min_periods=60).median()
        # soft probability via logistic of standardized distance to threshold
        sd = s.expanding(min_periods=60).std().replace(0, np.nan)
        z = ((s - thresh) / sd).fillna(0.0)
        p1 = 1 / (1 + np.exp(-z))
        return p1.fillna(self.base_p1_)

    def predict(self, X):
        p1 = self._regime_prob(X.index).values
        y0 = self.models_[0].predict(X[self.cols_].values)
        y1 = self.models_[1].predict(X[self.cols_].values)
        return (1 - p1) * y0 + p1 * y1

    def contributions(self, X):
        p1 = self._regime_prob(X.index).values[:, None]
        c0 = X[self.cols_].values * self.models_[0].coef_
        c1 = X[self.cols_].values * self.models_[1].coef_
        c = (1 - p1) * c0 + p1 * c1
        df = pd.DataFrame(c, index=X.index, columns=self.cols_)
        df["bias"] = ((1 - p1.ravel()) * self.models_[0].intercept_
                      + p1.ravel() * self.models_[1].intercept_)
        return df


def build_models(cfg: ModelConfig, regime_signal: pd.Series, rs=7):
    return {
        "ols": OLSModel(),
        "elastic_net": ElasticNetModel(cfg, rs),
        "xgboost": XGBModel(cfg, rs),
        "regime_switching": RegimeSwitchingModel(cfg, regime_signal, rs),
    }
