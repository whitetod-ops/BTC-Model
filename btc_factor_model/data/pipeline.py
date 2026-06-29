"""
Data pipeline (steps 3 & 8): align, lag, and build targets.

Two responsibilities, both about correctness:

1. ALIGNMENT  -- put every series on one daily business-day index, forward-fill
   slow series (macro/treasury) only up to their natural staleness, and assemble
   the raw panel.

2. LOOK-AHEAD SAFETY -- this is the heart of the whole exercise. Two distinct
   leakage channels are handled here:

   (a) Publication lag on FEATURES. A value timestamped at date D for, say, CPI
       or ETF flows was not actually *observable* until D + pub_lag_days. We
       shift every feature forward by its dictionary `pub_lag_days` so that the
       panel row at date D contains only information a trader knew at the close
       of D. On-chain/price are pub_lag 0; macro and filings are positive.

   (b) Forward-looking TARGETS. The targets are forward returns r_{t -> t+h}.
       Those are *supposed* to look ahead -- that is the prediction target. The
       danger is using them in training where they overlap the test window; that
       is handled later by the purged/embargoed walk-forward, not here.

The output is a single tidy frame: lagged features + one target column per
horizon, plus the (unlagged) price for plotting/attribution scaling.
"""
from __future__ import annotations
import pandas as pd

from ..config import HORIZONS
from .. import data_dictionary as dd


def align_panel(factors: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Put factors + price on one daily business-day grid."""
    idx = pd.bdate_range(min(factors.index.min(), price.index.min()),
                         max(factors.index.max(), price.index.max()))
    panel = factors.reindex(idx)
    panel["price"] = price.reindex(idx)
    # Forward-fill slow/stepwise series (macro releases, treasury filings,
    # weekly search) so daily rows are populated, but do NOT ffill price.
    slow_freqs = {"weekly", "monthly", "event"}
    var_freq = {v.id: v.frequency for v in dd.VARIABLES}
    for col in panel.columns:
        if col == "price":
            continue
        if var_freq.get(col) in slow_freqs:
            panel[col] = panel[col].ffill(limit=10)
    return panel


def apply_publication_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Shift each feature forward by its publication lag (look-ahead channel a).

    A positive shift means 'this number was only knowable this many days later',
    so row D ends up holding the most recent value that was actually published
    on or before D.
    """
    lags = dd.pub_lags()
    out = panel.copy()
    for col, lag in lags.items():
        if col in out.columns and lag > 0:
            out[col] = out[col].shift(lag)
    return out


def add_forward_return_targets(panel: pd.DataFrame,
                               price_col: str = "price") -> pd.DataFrame:
    """Add log forward-return targets r_{t->t+h} for every horizon.

    target_h[t] = log(P[t+h]) - log(P[t]).  These rows are only *scoreable*
    once t+h exists; the walk-forward harness enforces that no training target
    overlaps the test block via an embargo equal to the horizon.
    """
    import numpy as np
    out = panel.copy()
    logp = np.log(out[price_col])
    for name, h in HORIZONS.items():
        out[name] = logp.shift(-h) - logp
    return out


def build(factors: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Full pipeline: align -> lag features -> attach targets."""
    panel = align_panel(factors, price)
    panel = apply_publication_lags(panel)
    panel = add_forward_return_targets(panel)
    return panel


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """Raw feature ids present in the panel (excludes price + targets)."""
    targets = set(HORIZONS)
    ids = {v.id for v in dd.VARIABLES}
    return [c for c in panel.columns if c in ids and c not in targets]
