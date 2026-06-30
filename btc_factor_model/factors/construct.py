"""
Factor construction (steps -- derived & original factors).

This module builds the five headline original factors plus the supporting
derived series. Everything is causal. Inputs are assumed already
publication-lagged by data/pipeline.py, so derived factors inherit that safety.

ORIGINAL FACTORS
----------------
1. effective_tradable_float
2. derivative_notional_over_float
3. treasury_company_factor
4. power_law_deviation        (expanding-fit, no look-ahead)
5. funding_stress_factor      (USD/JPY + JPY vol carry-unwind)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")

# Static structural estimates (BTC). Tune to your best research.
LOST_COINS = 3_000_000           # provably lost / Patoshi dormant (Glassnode/Chainalysis range 3-4M)
GENESIS_SATOSHI = 1_100_000      # never-moved early coins
GOV_HOLDINGS = 200_000           # seized coins held by govts (US/CN/etc.), updateable


# --------------------------------------------------------------------------- #
# 1. EFFECTIVE TRADABLE FLOAT
# --------------------------------------------------------------------------- #
def effective_tradable_float(panel: pd.DataFrame) -> pd.Series:
    """Coins realistically available to trade.

        float = circulating
              - illiquid_supply        (low-spend-probability wallets)
              - treasury_holdings      (corporate, rarely sold)
              - etf_aum_btc            (ETF cold storage)
              - GOV_HOLDINGS           (seized, off-market)
              - (LOST + GENESIS)       (structurally gone)

    Double-count guard: Glassnode `illiquid_supply` already sweeps up some
    treasury/ETF/lost coins. We subtract treasury & ETF *beyond* illiquid using
    an overlap haircut so we don't remove the same coin twice. Tune `overlap`.
    """
    circ = panel["circulating_supply"]
    illiq = panel.get("illiquid_supply", pd.Series(0.0, index=panel.index)).fillna(0)
    treasury = panel.get("treasury_holdings", pd.Series(0.0, index=panel.index)).fillna(0)
    etf = panel.get("etf_aum_btc", pd.Series(0.0, index=panel.index)).fillna(0)

    overlap = 0.5  # fraction of treasury/ETF coins already inside `illiquid`
    structural_gone = LOST_COINS + GENESIS_SATOSHI + GOV_HOLDINGS

    flt = (circ
           - illiq
           - (1 - overlap) * (treasury + etf)
           - max(structural_gone - LOST_COINS, 0))  # lost already ~ illiquid

    return flt.clip(lower=circ * 0.05).rename("effective_float")


def float_pct(panel: pd.DataFrame, eff_float: pd.Series) -> pd.Series:
    return (eff_float / panel["circulating_supply"]).rename("float_pct")


# --------------------------------------------------------------------------- #
# 2. DERIVATIVE NOTIONAL / EFFECTIVE FLOAT
# --------------------------------------------------------------------------- #
def derivative_notional_over_float(panel: pd.DataFrame,
                                   eff_float: pd.Series) -> pd.Series:
    """Leverage-fragility ratio: USD derivative notional per tradable-coin of
    float (also in USD). High = thin spot float carrying a tall derivatives
    stack -> liquidation-cascade prone.

        ratio = (futures_oi_usd + options_oi_usd) * 1e9
                / (effective_float * price)
    """
    _zero = pd.Series(0.0, index=panel.index)
    deriv_usd = (panel.get("futures_oi_usd", _zero).fillna(0)
                 + panel.get("options_oi_usd", _zero).fillna(0)) * 1e9  # bn -> usd
    float_mktcap = eff_float * panel["price"]
    ratio = deriv_usd / float_mktcap.replace(0, np.nan)
    return ratio.rename("deriv_notional_over_float")


# --------------------------------------------------------------------------- #
# 3. TREASURY COMPANY FACTOR
# --------------------------------------------------------------------------- #
def treasury_company_factor(panel: pd.DataFrame) -> pd.Series:
    """Composite of corporate treasury demand. Blends:
        * accumulation pace (trailing 30d net buys, scaled by circ supply)
        * mNAV premium (reflexive issuance-funded-buying flywheel)
    Both are demand-positive, but extreme premium is reversal-prone -- that
    nonlinearity is left for the tree model / regime layer to exploit; here we
    keep a clean linear composite.
    """
    circ = panel["circulating_supply"]
    net_buys = panel.get("treasury_net_buys", pd.Series(0.0, index=panel.index)).fillna(0)
    mnav = panel.get("mstr_mnav", pd.Series(1.0, index=panel.index)).fillna(1.0)

    accel = (net_buys / circ)                      # demand as share of supply
    # standardize each leg with a causal expanding z then average
    def _czs(s):
        mu = s.expanding(min_periods=60).mean()
        sd = s.expanding(min_periods=60).std()
        return ((s - mu) / sd.replace(0, np.nan)).clip(-5, 5)
    factor = 0.6 * _czs(accel) + 0.4 * _czs(mnav)
    return factor.rename("treasury_company_factor")


# --------------------------------------------------------------------------- #
# 4. BITCOIN POWER-LAW DEVIATION  (expanding fit, no look-ahead)
# --------------------------------------------------------------------------- #
def power_law_fair_value(price: pd.Series, expanding: bool = True,
                         min_periods: int = 252) -> pd.Series:
    """Fitted power-law fair value via OLS of log(price) on log(days since
    genesis). When `expanding=True` the fit at date t uses ONLY data up to t,
    so the resulting deviation series is usable as a real-time feature with no
    look-ahead. `expanding=False` does a single full-sample fit (display only).
    """
    days = pd.Series((price.index - GENESIS).days.astype(float),
                     index=price.index)
    x = np.log(days)
    y = np.log(price)

    if not expanding:
        b, a = np.polyfit(x, y, 1)
        return pd.Series(np.exp(a + b * x), index=price.index, name="pl_fair")

    # Causal expanding OLS via running sums (O(n), no leakage).
    sx = x.expanding(min_periods).sum()
    sy = y.expanding(min_periods).sum()
    sxx = (x * x).expanding(min_periods).sum()
    sxy = (x * y).expanding(min_periods).sum()
    cnt = x.expanding(min_periods).count()
    denom = (cnt * sxx - sx * sx)
    b = (cnt * sxy - sx * sy) / denom.replace(0, np.nan)
    a = (sy - b * sx) / cnt
    fair = np.exp(a + b * x)
    return fair.rename("pl_fair")


def power_law_deviation(price: pd.Series, expanding: bool = True) -> pd.Series:
    fair = power_law_fair_value(price, expanding=expanding)
    return (np.log(price) - np.log(fair)).rename("power_law_dev")


# --------------------------------------------------------------------------- #
# 5. FUNDING STRESS FACTOR  (USD/JPY + JPY vol carry unwind)
# --------------------------------------------------------------------------- #
def funding_stress_factor(panel: pd.DataFrame,
                          window: int = 20) -> pd.Series:
    """Yen-carry-unwind stress. Carry unwinds show up as (i) yen appreciation
    (USD/JPY falling fast) and (ii) a JPY-vol spike, often together -- the
    Aug-2024 signature. Higher = more stress = BTC headwind.

        stress = z( -usdjpy_20d_return )      # yen strength
               + z(  jpy_vol_level )          # vol spike
               + z( -jpy_carry_diff_change )  # carry compression
    All z-scores are causal/expanding.
    """
    usdjpy = panel["usdjpy"]
    jpy_vol = panel.get("jpy_vol")
    carry = panel.get("jpy_carry_diff")

    yen_strength = -np.log(usdjpy).diff(window)          # + when yen strengthens
    carry_comp = -(carry.diff(window)) if carry is not None else None

    def _czs(s):
        if s is None:
            return pd.Series(0.0, index=panel.index)
        mu = s.expanding(min_periods=60).mean()
        sd = s.expanding(min_periods=60).std()
        return ((s - mu) / sd.replace(0, np.nan)).clip(-5, 5).fillna(0)

    stress = _czs(yen_strength) + _czs(jpy_vol) + 0.5 * _czs(carry_comp)
    return stress.rename("funding_stress_factor")


# --------------------------------------------------------------------------- #
# Assemble all derived/original factors onto the panel
# --------------------------------------------------------------------------- #
def add_original_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the original factors, building each only when its required inputs
    are present. This lets the real path go live incrementally (e.g. FRED+Yahoo
    before on-chain) without crashing on a not-yet-wired block; with the full
    panel (synthetic or all sources wired) every factor is built as before.
    """
    out = panel.copy()
    # effective_float is only meaningful WITH the dynamic dormant/illiquid (or
    # treasury/ETF) cohorts. With circulating supply ALONE it degenerates into a
    # monotone, near-deterministic supply trend that injects a spurious signal
    # (it contaminated the linear model on free data). Build the float factors
    # only when a real cohort input is present; otherwise omit them (honest > a
    # fake line). With the full panel every factor is built as before.
    cohorts = ("illiquid_supply", "lth_supply", "treasury_holdings", "etf_aum_btc")
    if "circulating_supply" in out.columns and any(c in out.columns for c in cohorts):
        eff = effective_tradable_float(out)
        out["effective_float"] = eff
        out["float_pct"] = float_pct(out, eff)
        if {"futures_oi_usd", "options_oi_usd"} & set(out.columns):
            out["deriv_notional_over_float"] = derivative_notional_over_float(out, eff)
    # Treasury factor needs actual treasury data (not just supply); else it is a
    # degenerate near-constant.
    if "circulating_supply" in out.columns and \
            ({"treasury_net_buys", "mstr_mnav"} & set(out.columns)):
        out["treasury_company_factor"] = treasury_company_factor(out)
    # Power-law deviation + price-momentum features (all price-derived, causal).
    if "price" in out.columns:
        out["power_law_dev"] = power_law_deviation(out["price"], expanding=True)
        p = out["price"].astype(float)
        lp = np.log(p)
        out["mom_30d"] = lp - lp.shift(21)     # ~1-month trend
        out["mom_90d"] = lp - lp.shift(63)     # ~3-month trend
        out["mom_180d"] = lp - lp.shift(126)   # ~6-month trend
        # (Mayer multiple removed: it is a trend gauge mislabeled as valuation and
        #  redundant with mom_180d -- see backtest.py --factors.)
    # Funding-stress factor needs USD/JPY.
    if "usdjpy" in out.columns:
        out["funding_stress_factor"] = funding_stress_factor(out)
    return out
