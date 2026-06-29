"""
Walk-forward validation (steps 7 & 8) -- the only validation used.

Design choices that kill look-ahead:

  * Origins march forward in time. Train on the past, test on the next block,
    advance, repeat. No fold ever trains on data later than its test block.

  * PURGING / EMBARGO. The target is an h-day forward return, so a training
    label stamped at date t actually 'knows about' prices out to t+h. If
    t + h >= test_start that label overlaps the test window and leaks. We drop
    every training row with t > test_start - h. The embargo equals the horizon
    -- longer horizons purge more, exactly as they should.

  * PCA FIT PER FOLD on training rows only, then applied to the test block.

  * Normalized features are causal rolling z-scores, safe to precompute once.

Outputs per (horizon, model): a continuous out-of-sample prediction series, the
matching out-of-sample contribution frame, and OOS metrics (R^2, rank IC,
sign hit-rate).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..config import WalkForwardConfig, PCAConfig, HORIZONS
from ..features.pca import CategoryPCA


@dataclass
class FoldResult:
    pred: pd.Series
    contrib: pd.DataFrame
    y_true: pd.Series


def _oos_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    d = pd.concat([y_true, y_pred], axis=1, keys=["y", "p"]).dropna()
    if len(d) < 10:
        return {"n": len(d), "r2": np.nan, "rank_ic": np.nan, "hit": np.nan}
    ss_res = ((d.y - d.p) ** 2).sum()
    ss_tot = ((d.y - d.y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    ic = spearmanr(d.y, d.p).correlation
    hit = (np.sign(d.y) == np.sign(d.p)).mean()
    return {"n": int(len(d)), "r2": float(r2), "rank_ic": float(ic),
            "hit": float(hit)}


class WalkForward:
    def __init__(self, wf: WalkForwardConfig, pca_cfg: PCAConfig):
        self.wf = wf
        self.pca_cfg = pca_cfg

    def _splits(self, n: int, embargo: int):
        """Yield (train_idx, test_idx) integer ranges with purging."""
        wf = self.wf
        start = wf.min_train
        origin = start
        while origin + wf.test_size <= n:
            test_lo, test_hi = origin, origin + wf.test_size
            train_hi = max(0, test_lo - embargo)          # purge overlap
            train_lo = 0 if wf.expanding else max(0, train_hi - wf.min_train)
            if train_hi - train_lo >= wf.min_train:
                yield (np.arange(train_lo, train_hi),
                       np.arange(test_lo, test_hi))
            origin += wf.step

    def run(self, Z_norm: pd.DataFrame, targets: pd.DataFrame,
            horizon_name: str, model_factory,
            pca_columns: list[str] | None = None) -> FoldResult:
        h = HORIZONS[horizon_name]
        y_all = targets[horizon_name]
        # common usable index: features present + target present
        common = Z_norm.dropna(how="all").index
        Z = Z_norm.reindex(common)
        y = y_all.reindex(common)
        n = len(common)

        preds, contribs = [], []
        for tr, te in self._splits(n, embargo=h):
            Z_tr_raw, Z_te_raw = Z.iloc[tr], Z.iloc[te]
            y_tr = y.iloc[tr]

            # rows with usable target in train
            keep = y_tr.notna()
            Z_tr_raw, y_tr = Z_tr_raw[keep], y_tr[keep]
            if len(y_tr) < self.wf.min_train // 2:
                continue

            # --- PCA fit on train only, transform both sides --------------
            pca = CategoryPCA(self.pca_cfg).fit(Z_tr_raw.fillna(0.0),
                                                pca_columns=pca_columns)
            F_tr = pca.transform(Z_tr_raw.fillna(0.0)).dropna()
            F_te = pca.transform(Z_te_raw.fillna(0.0))
            y_tr2 = y_tr.reindex(F_tr.index).dropna()
            F_tr = F_tr.reindex(y_tr2.index)
            if len(F_tr) < self.wf.min_train // 2:
                continue

            # --- fit model, predict OOS -----------------------------------
            model = model_factory()
            model.fit(F_tr, y_tr2)
            p = pd.Series(model.predict(F_te), index=F_te.index)
            c = model.contributions(F_te)
            preds.append(p)
            contribs.append(c)

        pred = (pd.concat(preds).sort_index() if preds
                else pd.Series(dtype=float))
        contrib = (pd.concat(contribs).sort_index() if contribs
                   else pd.DataFrame())
        return FoldResult(pred=pred, contrib=contrib,
                          y_true=y.reindex(pred.index))


def evaluate(result: FoldResult) -> dict:
    return _oos_metrics(result.y_true, result.pred)
