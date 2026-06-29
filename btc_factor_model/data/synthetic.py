"""
SYNTHETIC data generator  -- DEVELOPMENT / DEMO ONLY.
=====================================================

!!! THIS IS NOT REAL MARKET DATA. !!!

It exists so the entire pipeline (alignment -> normalization -> PCA -> models ->
walk-forward -> attribution -> dashboard) runs end-to-end without external API
access, and so unit tests are deterministic. The series are engineered to:

  * follow a Bitcoin-like power-law price trend with fat-tailed noise,
  * carry *modest, realistic* relationships to forward returns (so the models
    have something to find), and
  * match the column schema of the real adapters in data/sources.py exactly.

Numbers, levels, and especially any apparent predictive power are fabricated.
Swap in data/sources.py before drawing a single real conclusion.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")


def _ar1(n, rng, phi=0.97, scale=1.0, x0=0.0):
    """Persistent AR(1) latent driver -- macro/sentiment series are sticky."""
    x = np.empty(n)
    x[0] = x0
    eps = rng.normal(0, scale, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def generate(start="2018-01-01", end="2025-12-31", seed=7) -> dict:
    """Return dict(price=Series, factors=DataFrame) on a daily grid."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)            # business days
    n = len(idx)
    days_since_genesis = (idx - GENESIS).days.to_numpy().astype(float)

    # ----- latent macro / liquidity drivers --------------------------------
    liquidity = _ar1(n, rng, phi=0.995, scale=0.05)      # slow liquidity wave
    risk_off = _ar1(n, rng, phi=0.95, scale=0.30)        # risk appetite swings
    carry_stress = np.clip(_ar1(n, rng, phi=0.94, scale=0.35), -3, 6)

    # ----- power-law fair value + deviations -------------------------------
    # log10(price) ~ a + b*log10(days). Pick b so price spans a plausible range.
    b = 5.7
    a = -16.6
    log10_fair = a + b * np.log10(days_since_genesis)
    fair = 10 ** log10_fair

    # price deviation from fair value is a slow mean-reverting cycle + shocks,
    # nudged by liquidity (up) and risk_off / carry_stress (down)
    cycle = _ar1(n, rng, phi=0.985, scale=0.06)
    dev = (cycle
           + 0.6 * liquidity
           - 0.10 * risk_off
           - 0.06 * carry_stress)
    dev = dev - dev.mean()
    daily_noise = rng.standard_t(df=4, size=n) * 0.028   # fat tails
    log_price = np.log(fair) + dev + np.cumsum(daily_noise) * 0.0
    # blend: anchor to fair*exp(dev) but add realistic daily vol
    price = np.exp(np.log(fair) + dev) * np.exp(daily_noise)
    price = pd.Series(price, index=idx, name="price")

    fwd_ret_proxy = pd.Series(np.log(price)).diff().shift(-1).fillna(0).to_numpy()

    # power-law deviation factor (what the model will actually use)
    power_law_dev = np.log(price.to_numpy()) - np.log(fair)

    # ----- build factor columns (schema mirrors data/sources.py) -----------
    f = pd.DataFrame(index=idx)

    # macro_liquidity
    f["global_m2_usd"] = 90 + np.cumsum(0.002 + 0.01 * liquidity) + rng.normal(0, .02, n)
    f["fed_net_liquidity"] = 6.0 + 0.5 * liquidity + rng.normal(0, .03, n)
    f["dxy"] = 100 - 4 * liquidity + 2 * carry_stress * 0.3 + rng.normal(0, .4, n)
    f["real_rate_10y"] = 0.5 - 0.8 * liquidity + 0.2 * carry_stress + rng.normal(0, .05, n)
    f["move_index"] = 90 + 25 * np.clip(risk_off, 0, None) + 10 * np.abs(carry_stress) + rng.normal(0, 3, n)

    # funding_stress (ORIGINAL inputs)
    f["usdjpy"] = 130 - 6 * carry_stress + rng.normal(0, .8, n)   # stress = yen strength = lower usdjpy
    f["jpy_vol"] = 8 + 4 * np.clip(carry_stress, 0, None) + rng.normal(0, .6, n)
    f["jpy_carry_diff"] = 3.5 - 0.4 * carry_stress + rng.normal(0, .1, n)

    # risk_appetite
    f["vix"] = 16 + 9 * np.clip(risk_off, 0, None) + rng.normal(0, 1.2, n)
    f["hy_oas"] = 350 + 180 * np.clip(risk_off, 0, None) + rng.normal(0, 12, n)
    f["spx_mom"] = 0.01 - 0.05 * risk_off + rng.normal(0, .01, n)
    f["copper_gold"] = 0.20 - 0.02 * risk_off + rng.normal(0, .004, n)
    f["fear_greed"] = np.clip(50 + 18 * np.tanh(power_law_dev) - 15 * risk_off + rng.normal(0, 5, n), 1, 99)

    # etf_flows (zero before 2024-01-11 launch)
    launch = idx >= pd.Timestamp("2024-01-11")
    raw_flow = (300 * liquidity - 200 * risk_off + rng.normal(0, 120, n))
    f["etf_net_flow"] = np.where(launch, raw_flow, 0.0)
    f["etf_flow_5d"] = pd.Series(f["etf_net_flow"], index=idx).rolling(5).sum().fillna(0).to_numpy()
    f["etf_aum_btc"] = np.where(launch, np.cumsum(np.where(launch, raw_flow, 0)) / 60_000.0 + 600_000, np.nan)

    # derivatives
    f["perp_funding"] = 0.01 + 0.03 * np.tanh(power_law_dev) + rng.normal(0, .004, n)
    f["futures_oi_usd"] = 20 + 6 * np.tanh(power_law_dev) + 0.0008 * (price / price.iloc[0]) + rng.normal(0, 1.2, n)
    f["options_oi_usd"] = 18 + 5 * np.tanh(power_law_dev) + rng.normal(0, 1.0, n)
    f["annualized_basis"] = 5 + 6 * np.tanh(power_law_dev) - 3 * np.clip(risk_off, 0, None) + rng.normal(0, .8, n)
    f["put_call_ratio"] = 0.7 + 0.25 * np.clip(risk_off, 0, None) + rng.normal(0, .05, n)

    # onchain_supply
    circ = 16.6e6 + np.cumsum(np.full(n, 900.0)) - np.maximum.accumulate(np.zeros(n))
    circ = np.minimum(circ, 19.8e6)
    f["circulating_supply"] = circ
    illiq_frac = 0.74 + 0.03 * liquidity - 0.02 * risk_off + 0.01 * np.tanh(-power_law_dev)
    f["illiquid_supply"] = circ * np.clip(illiq_frac, 0.6, 0.85)
    f["lth_supply"] = circ * np.clip(0.66 + 0.03 * np.tanh(-power_law_dev) + 0.02 * liquidity, 0.55, 0.78)
    f["exchange_balance"] = circ * np.clip(0.12 - 0.02 * liquidity + 0.02 * risk_off, 0.07, 0.18)
    f["miner_balance"] = 1.8e6 - np.cumsum(np.full(n, 12.0)) + rng.normal(0, 2e3, n)

    # treasury_demand (ORIGINAL)
    accum = np.cumsum(np.clip(2500 + 4000 * liquidity + rng.normal(0, 800, n), 0, None))
    f["treasury_holdings"] = 150_000 + accum * (idx >= pd.Timestamp("2020-08-01"))
    f["treasury_net_buys"] = pd.Series(f["treasury_holdings"], index=idx).diff(30).fillna(0).to_numpy()
    f["mstr_mnav"] = np.clip(1.4 + 0.9 * np.tanh(power_law_dev) + 0.3 * liquidity + rng.normal(0, .08, n), 0.7, 4.0)

    # valuation (ORIGINAL power-law dev)
    f["power_law_dev"] = power_law_dev
    f["mvrv_z"] = 2.0 + 2.5 * np.tanh(power_law_dev) + rng.normal(0, .25, n)
    sma200 = pd.Series(price).rolling(200, min_periods=20).mean().to_numpy()
    f["mayer_multiple"] = price.to_numpy() / sma200
    f["nupl"] = np.clip(0.45 + 0.3 * np.tanh(power_law_dev) + rng.normal(0, .04, n), -0.3, 0.85)

    # regulatory (sparse signed events, exponentially decayed)
    reg = np.zeros(n)
    for ev_date, score in [("2024-01-11", +1.0), ("2021-09-24", -0.8),
                           ("2023-06-05", -0.6), ("2025-03-01", +0.5)]:
        d = pd.Timestamp(ev_date)
        if d in idx:
            pos = idx.get_loc(d)
            decay = score * np.exp(-np.arange(n - pos) / 30.0)
            reg[pos:] += decay
    f["reg_event_score"] = reg

    # adoption
    f["active_addresses"] = 900_000 * (1 + 0.4 * np.tanh(power_law_dev) + 0.2 * liquidity) + rng.normal(0, 1e4, n)
    f["stablecoin_supply"] = 80 + 60 * np.clip(np.cumsum(0.0005 + 0.002 * liquidity), 0, None) + rng.normal(0, 1, n)
    f["search_interest"] = np.clip(40 + 30 * np.tanh(power_law_dev) + rng.normal(0, 4, n), 1, 100)

    return {"price": price, "factors": f, "fair_value": pd.Series(fair, index=idx)}


if __name__ == "__main__":
    out = generate()
    print("SYNTHETIC price range:", out["price"].min().round(0),
          "->", out["price"].max().round(0))
    print("factor columns:", len(out["factors"].columns))
    print(out["factors"].tail(3).T.round(3))
