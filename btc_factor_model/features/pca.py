"""
PCA within parent factor categories (step 5).

Running one global PCA would blend macro liquidity with on-chain supply into
uninterpretable components. Instead we run a *separate* PCA inside each parent
category, keeping only enough components to explain `var_explained` (capped at
`max_components`). The resulting features look like:

    macro_liquidity_pc1, macro_liquidity_pc2, funding_stress_pc1, ...

Two correctness points:
  * FIT ON TRAIN ONLY. The transformer exposes sklearn-style fit/transform so
    the walk-forward harness fits PCA on each training fold and applies it to
    the held-out block -- never the reverse.
  * SIGN ALIGNMENT. PCA loadings have arbitrary sign. We flip each component so
    it points in the economically 'bullish' direction (positive correlation
    with the expected-sign-weighted average of its inputs). That makes a
    positive component value mean 'tailwind' everywhere, which is what keeps the
    later attribution table readable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..config import PCAConfig, PARENT_CATEGORIES
from .. import data_dictionary as dd


class CategoryPCA:
    def __init__(self, cfg: PCAConfig):
        self.cfg = cfg
        self.cat_cols: dict[str, list[str]] = {}
        self.models: dict[str, PCA] = {}
        self.signs: dict[str, np.ndarray] = {}
        self.passthrough: dict[str, list[str]] = {}
        self._standalone_cols: list[str] = []
        self.exp_sign = dd.expected_signs()
        self.out_columns_: list[str] = []

    def _category_map(self, columns: list[str]) -> dict[str, list[str]]:
        cat_of = {v.id: v.category for v in dd.VARIABLES}
        m: dict[str, list[str]] = {c: [] for c in PARENT_CATEGORIES}
        for col in columns:
            cat = cat_of.get(col)
            if cat in m:
                m[cat].append(col)
        return {k: v for k, v in m.items() if v}

    def fit(self, Z_train: pd.DataFrame,
            pca_columns: list[str] | None = None) -> "CategoryPCA":
        """Fit per-category PCA on `pca_columns` (default: all columns). Any
        column in Z_train NOT in pca_columns is treated as a standalone
        passthrough feature and emitted unchanged."""
        all_cols = list(Z_train.columns)
        pca_columns = list(pca_columns) if pca_columns is not None else all_cols
        self._standalone_cols = [c for c in all_cols if c not in pca_columns]
        self.cat_cols = self._category_map(pca_columns)
        self.models, self.signs, self.passthrough = {}, {}, {}
        self.out_columns_ = []
        for cat, cols in self.cat_cols.items():
            block = Z_train[cols].dropna()
            if len(cols) < self.cfg.min_features_for_pca or len(block) < 60:
                # not enough features/history -> pass features through as-is
                self.passthrough[cat] = cols
                self.out_columns_.extend(cols)
                continue
            n_comp = min(self.cfg.max_components, len(cols))
            pca = PCA(n_components=n_comp).fit(block.values)
            # choose how many PCs hit the variance target
            cum = np.cumsum(pca.explained_variance_ratio_)
            keep = int(np.searchsorted(cum, self.cfg.var_explained) + 1)
            keep = max(1, min(keep, n_comp))
            pca = PCA(n_components=keep).fit(block.values)

            # sign alignment: bullish-weighted composite of the block inputs
            w = np.array([self.exp_sign.get(c, 0) or 1 for c in cols], float)
            composite = block.values @ w
            scores = pca.transform(block.values)
            signs = np.ones(keep)
            for k in range(keep):
                cc = np.corrcoef(scores[:, k], composite)[0, 1]
                signs[k] = -1.0 if (cc == cc and cc < 0) else 1.0

            self.models[cat] = pca
            self.signs[cat] = signs
            self.out_columns_.extend([f"{cat}_pc{k+1}" for k in range(keep)])
        # standalone engineered/passthrough features go in by name, unchanged
        self.out_columns_.extend(self._standalone_cols)
        return self

    def transform(self, Z: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for cat, cols in self.cat_cols.items():
            if cat in self.passthrough:
                pieces.append(Z[cols])
                continue
            pca, signs = self.models[cat], self.signs[cat]
            vals = Z[cols].values
            scores = pca.transform(np.nan_to_num(vals, nan=0.0)) * signs
            names = [f"{cat}_pc{k+1}" for k in range(scores.shape[1])]
            pieces.append(pd.DataFrame(scores, index=Z.index, columns=names))
        if self._standalone_cols:
            pieces.append(Z[self._standalone_cols])
        out = pd.concat(pieces, axis=1)
        return out.reindex(columns=self.out_columns_)

    def fit_transform(self, Z_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(Z_train).transform(Z_train)

    def loadings(self) -> dict[str, pd.DataFrame]:
        """Per-category loading matrices (for interpreting components)."""
        out = {}
        for cat, pca in self.models.items():
            cols = self.cat_cols[cat]
            L = (pca.components_.T * self.signs[cat])
            out[cat] = pd.DataFrame(
                L, index=cols,
                columns=[f"{cat}_pc{k+1}" for k in range(L.shape[1])])
        return out
