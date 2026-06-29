"""
daily_run.py  -- one command that runs the whole model end to end
=================================================================

Builds the panel, constructs factors, runs the walk-forward models, and writes
the daily dashboard + the valuation cone. Works on synthetic data out of the box
(no API keys), and on real data once you wire data/sources.py.

    python -m btc_factor_model.daily_run                 # synthetic, all models
    python -m btc_factor_model.daily_run --source real   # once sources are wired
    python -m btc_factor_model.daily_run --models elastic_net xgboost --horizon fwd_60d

Outputs (in artifacts/):
    daily_dashboard.html   - today's read: factor scores, regime, prediction, attribution, residual
    valuation_cone.html    - upside / fair / downside cone with the regime-tilted path
    skill_metrics.csv      - rank IC / R2 / hit-rate for every model x horizon
    parent_scores.csv      - the 11 parent factor scores through time
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import SETTINGS, HORIZONS
from .data import synthetic, pipeline
from .factors.construct import add_original_factors
from .features.factor_engine import FactorEngine
from .models.model_engine import ModelEngine
from . import dashboard_daily
from .valuation_cone import ValuationCone, regime_score_from_parents, cone_chart

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def build_panel(source: str = "synthetic"):
    """Aligned panel with price, dictionary variables, the five originals, targets."""
    if source == "synthetic":
        syn = synthetic.generate()
        panel = pipeline.align_panel(syn["factors"], syn["price"])
    else:
        # REAL: implement the adapters in data/sources.py, then assemble here.
        from .data import sources
        factors = pipeline_real_factors(sources)   # you build this from the blocks
        price = sources.btc_price()
        panel = pipeline.align_panel(factors, price)
    # LOOK-AHEAD SAFETY (CLAUDE.md sec. 6): shift every feature forward by its
    # publication lag BEFORE deriving factors, so the originals inherit the lag.
    # Price is not a dictionary id, so targets below still use true price.
    panel = pipeline.apply_publication_lags(panel)
    panel = add_original_factors(panel)
    panel = pipeline.add_forward_return_targets(panel)
    return panel


def run(source="synthetic", models=("elastic_net", "xgboost"),
        horizon="fwd_60d", settings=SETTINGS):
    ARTIFACTS.mkdir(exist_ok=True)
    import warnings as _w, numpy as _np      # silence cosmetic PCA/divide warnings
    _w.filterwarnings("ignore", category=RuntimeWarning)
    _np.seterr(divide="ignore", invalid="ignore")
    print(f"[1/5] building panel ({source}) ...")
    panel = build_panel(source)
    print(f"      panel {panel.shape}  [{panel.index.min().date()} .. {panel.index.max().date()}]")

    print("[2/5] factor engine (normalize -> PCA -> parent scores) ...")
    eng = FactorEngine(settings)
    Z = eng.normalize(panel)
    eng.fit(Z)
    fs = eng.fit_transform(panel)
    fs.parent_scores.to_csv(ARTIFACTS / "parent_scores.csv")

    print(f"[3/5] model engine (walk-forward: {', '.join(models)} x {len(HORIZONS)} horizons) ...")
    me = ModelEngine(models, settings)
    res = me.run(Z[eng.universe_["all"]], panel[[h for h in HORIZONS if h in panel]],
                 pca_columns=eng.universe_["pca_columns"])
    res.metrics.to_csv(ARTIFACTS / "skill_metrics.csv", index=False)
    print(res.metrics.round(3).to_string(index=False))

    print("[4/5] valuation cone ...")
    rs = regime_score_from_parents(fs.parent_scores)
    cone = ValuationCone().run(panel["price"], regime_score=rs,
                               network=panel.get("active_addresses"))
    cpath = cone_chart(cone, out_path=str(ARTIFACTS / "valuation_cone.html"))
    t = cone.today
    print(f"      today ${t['price']:,.0f} vs fair ${t['fair']:,.0f} "
          f"({t['dev_pct']:+.0%}, {t['dev_pctile']:.0%}ile), regime {rs:+.2f}")
    print("      ->", cpath)

    print("[5/5] daily dashboard ...")
    model = models[0] if horizon else "xgboost"
    dpath = dashboard_daily.build_dashboard(
        panel, factor_scores=fs, model_results=res, cone=cone,
        model=(models[-1] if "xgboost" in models else models[0]), horizon=horizon,
        out_path=str(ARTIFACTS / "daily_dashboard.html"))
    print("      ->", dpath)
    print("\nDONE. Open the two HTML files in artifacts/.")
    return {"panel": panel, "factors": fs, "models": res, "cone": cone}


def pipeline_real_factors(sources):
    """Stitch the real source blocks into one factor frame. Implement once your
    data/sources.py adapters return data (see SETUP_DATA_FEEDS.md)."""
    import pandas as pd
    blocks = []
    for fn in (sources.macro_liquidity_block, sources.funding_stress_block,
               sources.derivatives_block, sources.onchain_block,
               sources.etf_flows_block, sources.treasury_block):
        try:
            blocks.append(fn())
        except NotImplementedError:
            print(f"      (skipping unimplemented source: {fn.__name__})")
        except Exception as e:  # a flaky/missing feed must not kill the whole run
            print(f"      (skipping {fn.__name__}: {type(e).__name__}: {e})")
    if not blocks:
        raise NotImplementedError("No real sources implemented yet — see SETUP_DATA_FEEDS.md")
    return pd.concat(blocks, axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the BTC factor model end to end.")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--models", nargs="+", default=["elastic_net", "xgboost"])
    ap.add_argument("--horizon", default="fwd_60d", choices=list(HORIZONS))
    a = ap.parse_args()
    run(source=a.source, models=tuple(a.models), horizon=a.horizon)
