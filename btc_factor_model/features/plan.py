"""
Feature plan: the explicit contract for what gets PCA'd vs. kept standalone.

Rationale. The five engineered original factors are already low-dimensional,
hand-built composites. Throwing them into a category PCA *with their own raw
ingredients* would double-count those ingredients and blur the attribution we
care most about. So:

  * RAW, independent variables  -> PCA within their parent category.
  * Engineered original factors -> kept standalone, never PCA'd, so each gets a
    clean, named line in every attribution table.
  * Variables that are pure inputs to an engineered factor (or near-
    deterministic, e.g. circulating supply) are EXCLUDED from PCA to avoid
    double counting.

The result is ~18-20 interpretable model features: a few category PCs, the five
originals by name, plus ETF-flow and regulatory passthroughs.
"""
from __future__ import annotations

# Engineered original factors -- always standalone, normalized but never PCA'd.
ENGINEERED_STANDALONE = [
    "effective_float",
    "deriv_notional_over_float",
    "treasury_company_factor",
    "power_law_dev",
    "funding_stress_factor",
]

# Raw variables excluded from PCA because they feed an engineered factor or are
# near-deterministic / redundant.
PCA_EXCLUDE = {
    # feed funding_stress_factor -> category represented by the factor itself
    "usdjpy", "jpy_vol", "jpy_carry_diff",
    # feed deriv_notional_over_float ratio
    "futures_oi_usd", "options_oi_usd",
    # feed treasury_company_factor
    "treasury_holdings", "treasury_net_buys", "mstr_mnav",
    # construction-only / redundant
    "circulating_supply", "float_pct", "etf_aum_btc",
}

# Raw passthroughs (kept as named features, too few to PCA usefully).
PASSTHROUGH_RAW = ["etf_net_flow", "etf_flow_5d", "reg_event_score"]


def modeling_universe(all_feature_cols: list[str]) -> dict:
    """Split the available columns into pca / standalone / passthrough groups."""
    standalone = [c for c in ENGINEERED_STANDALONE if c in all_feature_cols]
    passthrough = [c for c in PASSTHROUGH_RAW if c in all_feature_cols]
    pca_cols = [c for c in all_feature_cols
                if c not in PCA_EXCLUDE
                and c not in ENGINEERED_STANDALONE
                and c not in PASSTHROUGH_RAW]
    universe = sorted(set(standalone + passthrough + pca_cols))
    return {
        "pca_columns": pca_cols,
        "standalone": standalone,
        "passthrough": passthrough,
        "all": universe,
    }
