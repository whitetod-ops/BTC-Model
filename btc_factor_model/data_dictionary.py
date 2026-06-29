"""
DATA DICTIONARY (step 1)
========================

Single source of truth for every raw variable that enters the model. The
pipeline reads this registry to know what to pull, how to lag it (look-ahead
safety), what category it belongs to, and the economically-expected sign of its
effect on forward BTC returns.

Each Variable carries:
  id            : machine name (column in the aligned panel)
  name          : human label
  category      : one of config.PARENT_CATEGORIES
  source        : data vendor / endpoint family
  raw_field     : the vendor's native field name
  frequency     : native sampling frequency
  units         : units of the raw series
  pub_lag_days  : publication lag in days. THE most important column for
                  look-ahead avoidance -- the value stamped at date D was only
                  *knowable* at D + pub_lag_days, so the pipeline shifts it
                  forward by this many days before anything else happens.
  exp_sign      : expected sign of the factor's effect on FORWARD btc returns
                  (+1 bullish, -1 bearish, 0 ambiguous / regime-dependent)
  transform     : suggested pre-normalization transform
  notes         : construction caveats

Real source notes (you supply keys in data/sources.py):
  FRED        -> macro series (M2, WALCL, RRP, TGA, DXY, yields, HY OAS)
  Deribit/    -> options & futures OI, funding, basis
   Coinglass
  Glassnode   -> on-chain supply cohorts, MVRV, realized cap, addresses
  Farside/    -> US spot BTC ETF daily net flows
   issuers
  bitcointreasuries / filings -> corporate treasury holdings
  Yahoo/CME   -> USD/JPY, JPY vol proxy, equities, gold
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal
import pandas as pd


@dataclass(frozen=True)
class Variable:
    id: str
    name: str
    category: str
    source: str
    raw_field: str
    frequency: Literal["daily", "weekly", "monthly", "intraday", "event"]
    units: str
    pub_lag_days: int
    exp_sign: int
    transform: str
    notes: str = ""


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
VARIABLES: list[Variable] = [
    # ----- macro_liquidity --------------------------------------------------
    Variable("global_m2_usd", "Global M2 (USD proxy)", "macro_liquidity",
             "FRED+ECB+PBOC", "M2 aggregate", "weekly", "USD trn", 14, +1,
             "yoy_pct_change",
             "FX-converted sum of US/EU/JP/CN M2. Slow factor; drives 60d-12m."),
    Variable("fed_net_liquidity", "Fed net liquidity", "macro_liquidity",
             "FRED", "WALCL - WTREGEN - RRPONTSYD", "weekly", "USD trn", 1, +1,
             "level_then_diff",
             "Balance sheet minus TGA minus reverse repo. Daily-ish components."),
    Variable("dxy", "US Dollar Index", "macro_liquidity",
             "FRED/ICE", "DTWEXBGS or DXY", "daily", "index", 0, -1,
             "log_diff", "Inverse global-liquidity / risk proxy."),
    Variable("real_rate_10y", "10y TIPS real yield", "macro_liquidity",
             "FRED", "DFII10", "daily", "pct", 1, -1,
             "level", "Discount-rate channel; rising real rates pressure BTC."),
    Variable("move_index", "MOVE (rates vol)", "macro_liquidity",
             "ICE/BBG", "MOVE", "daily", "index", 0, -1,
             "level", "Rates volatility / macro stress."),

    # ----- funding_stress (ORIGINAL) ---------------------------------------
    Variable("usdjpy", "USD/JPY spot", "funding_stress",
             "Yahoo/FX", "JPY=X", "daily", "yen per usd", 0, +1,
             "log_diff",
             "Carry funder. Falling USDJPY = yen strength = carry unwind risk."),
    Variable("jpy_vol", "JPY implied vol", "funding_stress",
             "CME/BBG", "1m USDJPY ATM IV (or realized proxy)", "daily",
             "pct", 0, -1, "level",
             "Vol spike + yen appreciation = the Aug-2024 unwind signature."),
    Variable("jpy_carry_diff", "US-JP rate differential", "funding_stress",
             "FRED", "DGS2 - JP 2y", "daily", "pct", 1, +1,
             "level", "Width of the carry trade; compression = unwind pressure."),

    # ----- risk_appetite ----------------------------------------------------
    Variable("vix", "VIX", "risk_appetite",
             "CBOE/FRED", "VIXCLS", "daily", "index", 0, -1,
             "level", "Equity vol / risk-off. Contrarian only at extremes."),
    Variable("hy_oas", "US HY credit OAS", "risk_appetite",
             "FRED", "BAMLH0A0HYM2", "daily", "bps", 1, -1,
             "level", "Credit risk appetite; widening = risk-off."),
    Variable("spx_mom", "S&P 500 momentum", "risk_appetite",
             "Yahoo", "^GSPC 20d return", "daily", "pct", 0, +1,
             "level", "Risk-on beta. BTC retains high equity beta."),
    Variable("copper_gold", "Copper/Gold ratio", "risk_appetite",
             "Yahoo", "HG=F / GC=F", "daily", "ratio", 0, +1,
             "log_diff", "Growth vs safety cross-asset read."),
    Variable("fear_greed", "Crypto Fear & Greed", "risk_appetite",
             "alternative.me", "fng_value", "daily", "0-100", 0, +1,
             "level", "Native crypto sentiment; mean-reverting at tails."),

    # ----- etf_flows --------------------------------------------------------
    Variable("etf_net_flow", "US spot BTC ETF net flow", "etf_flows",
             "Farside/issuers", "sum daily net creations", "daily",
             "USD mn", 1, +1, "level",
             "Direct spot demand. T+1 reporting -> pub_lag 1."),
    Variable("etf_flow_5d", "ETF net flow (5d sum)", "etf_flows",
             "derived", "rolling 5d sum of etf_net_flow", "daily",
             "USD mn", 1, +1, "level", "Smoothed flow momentum."),
    Variable("etf_aum_btc", "ETF holdings in BTC", "etf_flows",
             "issuers", "sum of ETF BTC holdings", "daily", "BTC", 1, +1,
             "diff", "Stock of ETF-held coins; feeds effective float."),

    # ----- derivatives (ORIGINAL ratio lives here) -------------------------
    Variable("perp_funding", "Aggregate perp funding", "derivatives",
             "Coinglass", "oi-weighted funding rate", "intraday", "pct/8h",
             0, -1, "level",
             "Crowded-long detector. Very high funding = squeeze risk."),
    Variable("futures_oi_usd", "Futures+perp OI", "derivatives",
             "Coinglass", "aggregate open interest", "daily", "USD bn", 0, 0,
             "level", "Numerator of the notional/float ratio."),
    Variable("options_oi_usd", "Options OI notional", "derivatives",
             "Deribit", "options open interest", "daily", "USD bn", 0, 0,
             "level", "Adds to derivative notional stack."),
    Variable("annualized_basis", "3m futures basis", "derivatives",
             "Coinglass/CME", "annualized basis", "daily", "pct", 0, +1,
             "level", "Leverage demand / cash-and-carry richness."),
    Variable("put_call_ratio", "Options put/call", "derivatives",
             "Deribit", "volume put/call", "daily", "ratio", 0, -1,
             "level", "Hedging demand skew."),

    # ----- onchain_supply ---------------------------------------------------
    Variable("circulating_supply", "Circulating supply", "onchain_supply",
             "Glassnode", "supply_current", "daily", "BTC", 0, 0,
             "none", "Base for float construction; near-deterministic."),
    Variable("illiquid_supply", "Illiquid supply", "onchain_supply",
             "Glassnode", "supply_illiquid", "daily", "BTC", 1, +1,
             "diff", "Coins in wallets with low spend history. Squeeze fuel."),
    Variable("lth_supply", "Long-term holder supply", "onchain_supply",
             "Glassnode", "supply_lth (>155d)", "daily", "BTC", 1, +1,
             "diff", "Conviction holders; rising = accumulation regime."),
    Variable("exchange_balance", "Exchange balance", "onchain_supply",
             "Glassnode", "balance_exchanges", "daily", "BTC", 1, -1,
             "diff", "On-exchange = sellable. Falling balance is bullish."),
    Variable("miner_balance", "Miner balance", "onchain_supply",
             "Glassnode", "balance_miners", "daily", "BTC", 1, +1,
             "diff", "Miner distribution = latent sell pressure."),

    # ----- effective_float (ORIGINAL) --------------------------------------
    Variable("effective_float", "Effective tradable float", "effective_float",
             "derived", "see factors.construct.effective_tradable_float",
             "daily", "BTC", 1, -1,
             "level",
             "circulating - illiquid - treasury - ETF cold - gov - lost. "
             "Rising float = more sellable supply = headwind."),
    Variable("float_pct", "Float % of supply", "effective_float",
             "derived", "effective_float / circulating_supply", "daily",
             "pct", 1, -1, "level", "Scarcity gauge."),

    # ----- treasury_demand (ORIGINAL) --------------------------------------
    Variable("treasury_holdings", "Corp treasury holdings", "treasury_demand",
             "bitcointreasuries/filings", "sum public-co BTC", "event",
             "BTC", 2, +1, "diff",
             "MSTR/Strategy, Metaplanet, etc. Stepwise on 8-K filings."),
    Variable("treasury_net_buys", "Treasury net buys (30d)", "treasury_demand",
             "derived", "trailing 30d delta of holdings", "daily", "BTC",
             2, +1, "level", "Accumulation pace = demand flow."),
    Variable("mstr_mnav", "Treasury-co mNAV premium", "treasury_demand",
             "market", "mkt cap / BTC NAV", "daily", "x", 0, +1,
             "level",
             "Premium enables issuance-funded buying (reflexive). Extreme "
             "premium = reversal risk; treat as regime-dependent at tails."),

    # ----- valuation (ORIGINAL power-law deviation lives here) -------------
    Variable("power_law_dev", "Power-law deviation", "valuation",
             "derived", "log(price) - log(powerlaw_fair)", "daily", "log",
             1, -1, "level",
             "Distance above/below fitted time^n trend. Mean-reverts at 6-12m."),
    Variable("mvrv_z", "MVRV Z-score", "valuation",
             "Glassnode", "mvrv_z_score", "daily", "z", 1, -1,
             "level", "Market vs realized cap; >7 historically a top zone."),
    Variable("mayer_multiple", "Mayer multiple", "valuation",
             "derived", "price / 200d SMA", "daily", "x", 0, -1,
             "level", "Trend stretch; >2.4 historically overheated."),
    Variable("nupl", "Net unrealized P/L", "valuation",
             "Glassnode", "nupl", "daily", "ratio", 1, -1,
             "level", "Aggregate paper gains; euphoria gauge."),

    # ----- regulatory -------------------------------------------------------
    Variable("reg_event_score", "Regulatory event score", "regulatory",
             "curated/NLP", "signed event impact", "event", "score",
             0, 0, "decay",
             "Event-study dummies (ETF approval +, enforcement -) with "
             "exponential decay. Sign supplied per event."),

    # ----- adoption ---------------------------------------------------------
    Variable("active_addresses", "Active addresses", "adoption",
             "Glassnode", "active_addr", "daily", "count", 1, +1,
             "yoy_pct_change", "Network usage / adoption trend."),
    Variable("stablecoin_supply", "Stablecoin supply", "adoption",
             "Glassnode/DefiLlama", "USDT+USDC+ supply", "daily", "USD bn",
             1, +1, "yoy_pct_change", "Crypto-native dry powder."),
    Variable("search_interest", "Search interest", "adoption",
             "Google Trends", "bitcoin query index", "weekly", "0-100", 3, +1,
             "level", "Retail attention; lags price, use carefully."),

    # ----- momentum (price-derived trend; computed in factors.construct) ----
    Variable("mom_30d", "1-month price momentum", "momentum",
             "derived", "log return over ~21 trading days", "daily", "log",
             0, +1, "level", "Short trend; persists over 1-3m."),
    Variable("mom_90d", "3-month price momentum", "momentum",
             "derived", "log return over ~63 trading days", "daily", "log",
             0, +1, "level", "Medium trend; classic 1-3m predictor."),
    Variable("mom_180d", "6-month price momentum", "momentum",
             "derived", "log return over ~126 trading days", "daily", "log",
             0, +1, "level", "Half-year trend; fades/reverses at 12m+."),
]


def as_frame() -> pd.DataFrame:
    """Return the dictionary as a tidy DataFrame (for export / inspection)."""
    df = pd.DataFrame([asdict(v) for v in VARIABLES])
    return df[
        ["id", "name", "category", "source", "raw_field", "frequency",
         "units", "pub_lag_days", "exp_sign", "transform", "notes"]
    ]


def by_category() -> dict[str, list[Variable]]:
    out: dict[str, list[Variable]] = {}
    for v in VARIABLES:
        out.setdefault(v.category, []).append(v)
    return out


def expected_signs() -> dict[str, int]:
    return {v.id: v.exp_sign for v in VARIABLES}


def pub_lags() -> dict[str, int]:
    return {v.id: v.pub_lag_days for v in VARIABLES}


if __name__ == "__main__":
    df = as_frame()
    out = ARTIFACTS_CSV = "data_dictionary.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} variables across {df.category.nunique()} categories "
          f"-> {out}")
    print(df.groupby("category").id.count().rename("n_vars"))
