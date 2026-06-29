"""
End-to-end driver.

Wires the whole chain together for every (horizon, model) pair:

    raw data -> align + publication-lag (look-ahead safe)
             -> construct original/derived factors
             -> causal rolling z-score normalization
             -> [per fold] category PCA -> model fit -> OOS predict + contribs
             -> OOS metrics + factor attribution tables

Run with synthetic data out of the box; flip `source='real'` once the adapters
in data/sources.py have keys.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import warnings
import numpy as np
import pandas as pd

from .config import SETTINGS, HORIZONS
from .data import synthetic, pipeline
from .factors import construct, transforms
from .features import normalize, plan
from .features.pca import CategoryPCA
from .models.walk_forward import WalkForward, evaluate
from .models.estimators import build_models
from .attribution import contributions as attrib
from . import data_dictionary as dd

warnings.filterwarnings("ignore")


@dataclass
class Results:
    panel: pd.DataFrame
    Z: pd.DataFrame
    universe: dict
    metrics: pd.DataFrame
    fold_results: dict = field(default_factory=dict)   # (model,horizon)->FoldResult
    attribution: dict = field(default_factory=dict)
    fair_value: pd.Series = None


def _build_panel(source: str) -> tuple[pd.DataFrame, pd.Series]:
    if source == "synthetic":
        gen = synthetic.generate()
        panel = pipeline.build(gen["factors"], gen["price"])
        return panel, gen.get("fair_value")
    elif source == "real":
        from .data import sources
        # assemble blocks; user wires these up
        blocks = [b() for b in sources.REAL_BLOCKS]
        factors = pd.concat(blocks, axis=1)
        price = sources.btc_price()
        panel = pipeline.build(factors, price)
        return panel, None
    raise ValueError(source)


def _normalized_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # 1. apply the dictionary's pre-normalization transforms to raw features
    transformed = pd.DataFrame(index=panel.index)
    tmap = {v.id: v.transform for v in dd.VARIABLES}
    for col in pipeline.feature_columns(panel):
        if col in panel.columns:
            transformed[col] = transforms.apply_transform(panel[col], tmap.get(col, "level"))
    # engineered originals are already on sensible scales; pass raw level in
    for col in plan.ENGINEERED_STANDALONE:
        if col in panel.columns:
            transformed[col] = panel[col]

    universe = plan.modeling_universe(list(transformed.columns))
    cols = universe["all"]
    # 2. causal rolling z-score normalization
    Z = normalize.normalize_frame(transformed, cols, SETTINGS.norm)
    return Z, universe


def run(source: str = "synthetic", verbose: bool = True) -> Results:
    panel, fair = _build_panel(source)
    if verbose:
        print(f"Panel: {panel.shape[0]} rows x {panel.shape[1]} cols "
              f"[{panel.index.min().date()} .. {panel.index.max().date()}]")

    # original + derived factors (look-ahead-safe construction)
    panel = construct.add_original_factors(panel)
    # rebuild targets after adding engineered factors (price unchanged)
    panel = pipeline.add_forward_return_targets(panel)

    Z, universe = _normalized_features(panel)
    if verbose:
        print(f"Model features ({len(universe['all'])}): "
              f"{universe['pca_columns'][:4]}... + originals "
              f"{universe['standalone']}")

    targets = panel[list(HORIZONS)]
    # regime signal: trailing 20d realized vol of BTC log returns (causal)
    rets = np.log(panel["price"]).diff()
    regime_signal = rets.rolling(20).std()

    wf = WalkForward(SETTINGS.wf, SETTINGS.pca)
    rows, fold_results, attribution = [], {}, {}

    for hname in HORIZONS:
        models = build_models(SETTINGS.model, regime_signal, SETTINGS.random_state)
        for mname, _ in models.items():
            def factory(mn=mname):
                return build_models(SETTINGS.model, regime_signal,
                                    SETTINGS.random_state)[mn]
            res = wf.run(Z, targets, hname, factory,
                         pca_columns=universe["pca_columns"])
            m = evaluate(res)
            m.update({"horizon": hname, "model": mname})
            rows.append(m)
            fold_results[(mname, hname)] = res
            if not res.contrib.empty:
                attribution[(mname, hname)] = {
                    "feature": attrib.feature_contribution_table(res.contrib),
                    "category": attrib.category_contribution_table(res.contrib),
                    "latest": attrib.latest_decomposition(res.contrib, res.pred),
                    "reconciliation": attrib.reconciliation_check(res.contrib, res.pred),
                }
            if verbose:
                print(f"  {hname:7s} {mname:16s} "
                      f"R2={m['r2']:+.3f} IC={m['rank_ic']:+.3f} "
                      f"hit={m['hit']:.2f} n={m['n']}")

    metrics = pd.DataFrame(rows)
    return Results(panel=panel, Z=Z, universe=universe, metrics=metrics,
                   fold_results=fold_results, attribution=attribution,
                   fair_value=fair)


if __name__ == "__main__":
    run()
