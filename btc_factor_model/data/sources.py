"""
Data source adapters (step 2/3 -- ingestion).

Each adapter returns a tidy daily DataFrame indexed by date with columns whose
names match `id`s in the data dictionary. The pipeline does NOT care where the
data came from -- it only needs the contract:

    adapter() -> pd.DataFrame  (DatetimeIndex, columns = subset of dictionary ids)

WHAT IS WIRED (free stack):
    btc_price              Coin Metrics PriceUSD (2010+) + yfinance (current) -> price
                           + Stooq fallback
    macro_liquidity_block  FRED (net liquidity, DXY, real rate, MOVE, M2,    -> macro_liquidity
                           VIX, HY OAS) + Yahoo (S&P mom, copper/gold)          + risk_appetite
                           + Fear & Greed (alternative.me)
    funding_stress_block   FRED/Yahoo USD/JPY + carry diff + realized-vol     -> funding_stress
                           JPY-vol proxy
    onchain_block          Coin Metrics community: supply, active addrs,      -> onchain_supply,
                           causal MVRV z, dormant-supply illiquid/LTH proxy,     valuation,
                           + stablecoin supply (DefiLlama)                       adoption
    etf_flows_block        Farside spot-ETF daily net flow (+5d sum)          -> etf_flows

STILL STUBS (raise NotImplementedError -> pipeline_real_factors skips them):
    derivatives_block, treasury_block, regulatory_events
    (see SETUP_DATA_FEEDS.md; a flaky live feed is also skipped, not fatal)

KEYS / NETWORK
--------------
FRED needs FRED_API_KEY (free, 32-char) in a `.env` file at the project root;
this module loads it automatically via python-dotenv. Yahoo/Stooq need no key.
Live pulls require outbound network to api.stlouisfed.org / query*.finance.yahoo.com
/ stooq.com -- run on a machine that can reach them (a locked-down sandbox cannot,
in which case the synthetic source in data/synthetic.py is the offline path).

LOOK-AHEAD: adapters return data stamped at its OBSERVATION date. Do NOT apply
publication lags here -- the pipeline does that once (pipeline.apply_publication_lags).
"""
from __future__ import annotations
import os
import warnings

import numpy as np
import pandas as pd

# Load FRED_API_KEY (and friends) from a project-root .env if present.
try:  # optional dependency; silently skip if unavailable
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from . import macro_data  # FRED/Yahoo macro block (handles the unit gotcha)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _require_fred(block: str) -> None:
    """Fail loudly (not as 'unimplemented') when a wired FRED block has no key."""
    if "FRED_API_KEY" not in os.environ:
        raise RuntimeError(
            f"{block} is wired for real FRED data but FRED_API_KEY is not set. "
            "Add it to a .env file at the project root (see SETUP_DATA_FEEDS.md), "
            "or run with --source synthetic."
        )


def _yahoo_close(ticker: str, start: str) -> pd.Series:
    """Daily close for one Yahoo ticker. Raises on empty/failure."""
    import yfinance as yf
    d = yf.download(ticker, start=start, interval="1d",
                    auto_adjust=False, progress=False, threads=False)
    if d is None or len(d) == 0:
        raise RuntimeError(f"yfinance returned no rows for {ticker!r}.")
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d["Close"].astype(float).rename(ticker)


def _to_bgrid(df: pd.DataFrame, *, ffill_limit: int = 7,
              slow_cols: tuple = ()) -> pd.DataFrame:
    """Reindex to a business-day grid; ffill daily gaps lightly and slow
    (monthly/weekly) series generously so daily rows are populated."""
    if df.empty:
        return df
    bidx = pd.bdate_range(df.index.min(), df.index.max())
    out = df.reindex(bidx)
    for col in out.columns:
        limit = None if col in slow_cols else ffill_limit
        out[col] = out[col].ffill(limit=limit)
    return out


# --------------------------------------------------------------------------- #
# Macro (FRED) -- single series passthrough kept for ad-hoc use
# --------------------------------------------------------------------------- #
def fred_series(series_id: str, start: str = "2017-01-01") -> pd.Series:
    """Pull a single FRED series (observation-dated). Requires FRED_API_KEY."""
    if "FRED_API_KEY" not in os.environ:
        raise NotImplementedError(
            "Set FRED_API_KEY and wire up fredapi here. "
            "FRED ids: M2SL, WALCL, WTREGEN, RRPONTSYD, DTWEXBGS, DFII10, "
            "VIXCLS, BAMLH0A0HYM2, DGS2."
        )
    from fredapi import Fred  # noqa
    s = Fred(api_key=os.environ["FRED_API_KEY"]).get_series(
        series_id, observation_start=start)
    return pd.Series(s, name=series_id)


# --------------------------------------------------------------------------- #
# Macro liquidity + risk appetite (FRED + Yahoo)
# --------------------------------------------------------------------------- #
def macro_liquidity_block(start: str = "2017-01-01",
                          source: str = "fred") -> pd.DataFrame:
    """FRED-driven macro block + Yahoo cross-asset risk reads.

    Returns dictionary ids -- macro_liquidity: fed_net_liquidity, dxy,
    real_rate_10y, move_index, global_m2_usd; risk_appetite: vix, hy_oas,
    spx_mom, copper_gold.

    The FRED math (incl. the net-liquidity unit gotcha) lives in macro_data; we
    select the columns this block owns and add M2 + Yahoo extras. fear_greed
    (alternative.me) and the rest of risk_appetite are left for a later wire-up.
    """
    if source == "fred":
        _require_fred("macro_liquidity_block")

    # FRED-derived columns (daily business-day grid already).
    m = macro_data.macro_block(source=source, start=start)
    own = ["fed_net_liquidity", "dxy", "real_rate_10y", "move_index",
           "vix", "hy_oas"]
    out = m[[c for c in own if c in m.columns]].copy()

    # Global M2 (USD). FRED M2SL is US M2 in $bn -> $trn. Documented as a
    # US-only PROXY for the dictionary's FX-converted global aggregate; swap in
    # ECB/PBOC components later for the full measure.
    try:
        if source in ("fred", "auto"):
            m2 = fred_series("M2SL", start)              # $bn, monthly
            out = out.join((m2 / 1_000.0).rename("global_m2_usd"), how="outer")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"global_m2_usd (M2SL) unavailable ({type(e).__name__}); "
                      "column omitted.", stacklevel=2)

    # Yahoo cross-asset risk reads (skip entirely on the offline synthetic path).
    if source != "synthetic":
        try:
            spx = _yahoo_close("^GSPC", start)
            out = out.join(np.log(spx).diff(20).rename("spx_mom"), how="outer")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"spx_mom (^GSPC) unavailable ({type(e).__name__}); "
                          "column omitted.", stacklevel=2)
        try:
            copper = _yahoo_close("HG=F", start)
            gold = _yahoo_close("GC=F", start)
            cg = (copper / gold).rename("copper_gold")
            out = out.join(cg, how="outer")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"copper_gold (HG=F/GC=F) unavailable "
                          f"({type(e).__name__}); column omitted.", stacklevel=2)
        try:
            import requests
            data = requests.get("https://api.alternative.me/fng/",
                                params={"limit": 0}, timeout=30).json()["data"]
            fg = pd.Series({
                pd.to_datetime(int(x["timestamp"]), unit="s").normalize(): float(x["value"])
                for x in data}).sort_index()
            out = out.join(fg.rename("fear_greed"), how="outer")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"fear_greed (alternative.me) unavailable "
                          f"({type(e).__name__}); column omitted.", stacklevel=2)

    return _to_bgrid(out, slow_cols=("global_m2_usd",))


# --------------------------------------------------------------------------- #
# FX / funding stress (FRED USD/JPY + carry, realized-vol JPY-vol proxy)
# --------------------------------------------------------------------------- #
def funding_stress_block(start: str = "2017-01-01",
                         source: str = "fred") -> pd.DataFrame:
    """usdjpy, jpy_carry_diff, jpy_vol.

    usdjpy + jpy_carry_diff come from macro_data (FRED DEXJPUS / DGS2 - JP2y).
    jpy_vol: 1m USD/JPY ATM IV is not free, so we use the 20d realized vol of
    USD/JPY log returns (annualized, %) as a documented free proxy.
    """
    if source == "fred":
        _require_fred("funding_stress_block")

    m = macro_data.macro_block(source=source, start=start)
    out = pd.DataFrame(index=m.index)
    if "usdjpy" in m.columns:
        out["usdjpy"] = m["usdjpy"]
    if "jpy_carry_diff" in m.columns:
        out["jpy_carry_diff"] = m["jpy_carry_diff"]

    if "usdjpy" in m.columns:
        r = np.log(m["usdjpy"].astype(float)).diff()
        out["jpy_vol"] = (r.rolling(20).std() * np.sqrt(252) * 100.0)

    return _to_bgrid(out)


# --------------------------------------------------------------------------- #
# Derivatives (Coinglass / Deribit) -- STUB
# --------------------------------------------------------------------------- #
def derivatives_block(start: str = "2017-01-01") -> pd.DataFrame:
    """perp_funding, futures_oi_usd, options_oi_usd, annualized_basis,
    put_call_ratio.  Coinglass v4 + Deribit public API. See SETUP_DATA_FEEDS.md
    sec. 3 (US geo caveat -- de-weight at first)."""
    raise NotImplementedError("Wire up Coinglass/Deribit for derivatives.")


# --------------------------------------------------------------------------- #
# On-chain (Coin Metrics community -- free, no key)
# --------------------------------------------------------------------------- #
def _cm_fetch(client, metrics, start: str) -> pd.DataFrame:
    """Fetch Coin Metrics asset metrics one at a time so an unavailable
    (paid-tier) metric is skipped rather than failing the whole pull."""
    frames = []
    for m in metrics:
        try:
            d = client.get_asset_metrics(assets="btc", metrics=[m],
                                         frequency="1d",
                                         start_time=start).to_dataframe()
            t = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None).dt.normalize()
            s = pd.to_numeric(d[m], errors="coerce")
            s.index = t.values
            frames.append(s.rename(m))
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Coin Metrics metric {m} unavailable "
                          f"({type(e).__name__}); skipped.", stacklevel=2)
    if not frames:
        raise RuntimeError("Coin Metrics returned no usable metrics.")
    out = pd.concat(frames, axis=1)
    return out[~out.index.duplicated(keep="last")].sort_index()


def onchain_block(start: str = "2017-01-01") -> pd.DataFrame:
    """On-chain block from Coin Metrics Community (free, no key).

    Wired (community tier): circulating_supply (SplyCur), active_addresses
    (AdrActCnt), and a causal MVRV z-score (mvrv_z) built from market cap
    (CapMrktCurUSD) over realized cap (CapRealUSD) -- expanding mean/std only,
    so it is real-time safe.

    NOT free on Coin Metrics community: the Glassnode supply cohorts
    (illiquid_supply, lth_supply, exchange_balance, miner_balance) and nupl.
    They are omitted -> effective_float falls back to circulating supply minus
    structural sinks (coarser, but live). Add Glassnode later for the cohorts.
    """
    try:
        from coinmetrics.api_client import CoinMetricsClient
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "onchain_block needs the Coin Metrics client: "
            "pip install coinmetrics-api-client"
        ) from e
    client = CoinMetricsClient()  # community tier (no API key)

    raw = _cm_fetch(client, ["SplyCur", "AdrActCnt", "CapMrktCurUSD",
                             "CapRealUSD", "SplyAct1yr", "SplyAct180d",
                             "SplyActPct1yr"], start)
    out = pd.DataFrame(index=raw.index)
    if "SplyCur" in raw.columns:
        out["circulating_supply"] = raw["SplyCur"]
    if "AdrActCnt" in raw.columns:
        out["active_addresses"] = raw["AdrActCnt"]
    if {"CapMrktCurUSD", "CapRealUSD"}.issubset(raw.columns):
        mvrv = raw["CapMrktCurUSD"] / raw["CapRealUSD"].replace(0, np.nan)
        mu = mvrv.expanding(min_periods=180).mean()
        sd = mvrv.expanding(min_periods=180).std()
        out["mvrv_z"] = ((mvrv - mu) / sd.replace(0, np.nan)).clip(-5, 5)

    # Dormant / HODLed supply -> free proxy for the locked-up cohorts that
    # effective_float subtracts. "Supply active in last X" taken away from
    # current supply = coins that have NOT moved in X: the illiquid (>1yr) and
    # long-term-holder (>180d) buckets. Prefer the absolute SplyActNyr metrics;
    # fall back to the percentage metric. This is the scarcity signal -- coins
    # going dormant (bullish) vs waking up onto exchanges (bearish).
    sply = raw.get("SplyCur")
    if sply is not None:
        if "SplyAct1yr" in raw.columns:
            out["illiquid_supply"] = (sply - raw["SplyAct1yr"]).clip(lower=0)
        elif "SplyActPct1yr" in raw.columns:
            pct = raw["SplyActPct1yr"]
            pct = pct / 100.0 if float(pct.max()) > 1.5 else pct
            out["illiquid_supply"] = (sply * (1.0 - pct)).clip(lower=0)
        if "SplyAct180d" in raw.columns:
            out["lth_supply"] = (sply - raw["SplyAct180d"]).clip(lower=0)
        elif "illiquid_supply" in out.columns:
            out["lth_supply"] = out["illiquid_supply"]

    # Stablecoin supply (DefiLlama, free) -> adoption "dry powder" entering crypto.
    try:
        import requests
        j = requests.get("https://stablecoins.llama.fi/stablecoincharts/all",
                         timeout=30).json()
        sc = pd.Series({
            pd.to_datetime(int(d["date"]), unit="s").normalize():
                float(d["totalCirculatingUSD"]["peggedUSD"])
            for d in j if d.get("totalCirculatingUSD")}).sort_index() / 1e9
        out = out.join(sc.rename("stablecoin_supply"), how="outer")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"stablecoin_supply (DefiLlama) unavailable "
                      f"({type(e).__name__}); column omitted.", stacklevel=2)

    return _to_bgrid(out)


# --------------------------------------------------------------------------- #
# ETF flows (Farside scrape -- free)
# --------------------------------------------------------------------------- #
def etf_flows_block(start: str = "2024-01-11") -> pd.DataFrame:
    """US spot BTC ETF net flows from Farside (free). Returns etf_net_flow
    (USD mn) and etf_flow_5d (rolling 5d sum). The complex launched 2024-01-11;
    earlier dates are structurally absent. A User-Agent avoids a 403."""
    import io
    import requests
    html = requests.get("https://farside.co.uk/bitcoin-etf-flow-all-data/",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    raw = pd.read_html(io.StringIO(html))[0].copy()
    raw.columns = [str(c[-1]) if isinstance(c, tuple) else str(c)
                   for c in raw.columns]
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).set_index("date")
    if "Total" not in raw.columns:
        raise RuntimeError("Farside table has no 'Total' column; layout changed.")
    # Farside cells: "1,234.5", negatives in "(1,234.5)", blanks as "-".
    s = raw["Total"].astype(str).str.strip().str.replace(",", "", regex=False)
    neg = s.str.startswith("(")
    s = s.str.replace(r"[()]", "", regex=True).replace(
        {"-": None, "": None, "nan": None})
    v = pd.to_numeric(s, errors="coerce")
    v[neg] = -v[neg].abs()
    out = pd.DataFrame(index=raw.index)
    out["etf_net_flow"] = v
    out["etf_flow_5d"] = out["etf_net_flow"].rolling(5, min_periods=1).sum()
    out = out[out.index >= pd.Timestamp(start)].dropna(how="all")
    return _to_bgrid(out)


# --------------------------------------------------------------------------- #
# Treasury companies (filings / bitcointreasuries) -- STUB
# --------------------------------------------------------------------------- #
def treasury_block(start: str = "2020-08-01") -> pd.DataFrame:
    """treasury_holdings, treasury_net_buys (derived), mstr_mnav.
    Holdings step on 8-K filings; forward-fill between events."""
    raise NotImplementedError("Wire up treasury holdings source.")


# --------------------------------------------------------------------------- #
# Price (the target)
# --------------------------------------------------------------------------- #
def _coinmetrics_price(start: str = "2010-07-18") -> pd.Series:
    """BTC/USD daily close from Coin Metrics community PriceUSD -- history back
    to 2010 (free, no key). This is the deep history the power-law fit needs."""
    from coinmetrics.api_client import CoinMetricsClient
    raw = _cm_fetch(CoinMetricsClient(), ["PriceUSD"], start)
    return raw["PriceUSD"].astype(float).rename("price")


def btc_price(start: str = "2010-07-18") -> pd.Series:
    """Daily BTC/USD close with the LONGEST free history available.

    Combines Coin Metrics PriceUSD (back to 2010) with yfinance 'BTC-USD'
    (current), yfinance taking priority where the two overlap so the recent end
    is unchanged; Coin Metrics fills the deep 2010-2014 history that yfinance
    lacks. Falls back to Stooq if both fail. The extra early years stabilize the
    power-law fit and add independent long-horizon windows for validation."""
    from . import btc_returns
    parts = []
    try:                                            # deep history (2010+)
        parts.append(_coinmetrics_price(start))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Coin Metrics PriceUSD unavailable ({type(e).__name__}); "
                      "deep history skipped.", stacklevel=2)
    try:                                            # current end (priority)
        yf = btc_returns.load_ohlcv(source="yfinance", start="2014-01-01")
        parts.append(yf["close"].astype(float).rename("price"))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"yfinance BTC-USD failed ({type(e).__name__}).",
                      stacklevel=2)
    if not parts:
        return _stooq_btc(start)
    combined = parts[0]
    for nxt in parts[1:]:                           # later source wins on overlap
        combined = nxt.combine_first(combined)
    combined = combined.sort_index()
    return combined[combined.index >= pd.Timestamp(start)].rename("price")


def _stooq_btc(start: str = "2017-01-01") -> pd.Series:
    """Backup BTC/USD daily close from Stooq (no key)."""
    import io
    import requests
    url = "https://stooq.com/q/d/l/?s=btcusd&i=d"
    txt = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                       timeout=30).text
    df = pd.read_csv(io.StringIO(txt))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    s = df["Close"].astype(float).rename("price")
    return s[s.index >= pd.Timestamp(start)]


def regulatory_events(start: str = "2017-01-01") -> pd.DataFrame:
    """reg_event_score: a curated, signed, dated event table decayed to daily.
    Columns: date, score (signed magnitude). See factors.construct.decay_events.
    """
    raise NotImplementedError("Supply curated regulatory event table.")


REAL_BLOCKS = [
    macro_liquidity_block, funding_stress_block, derivatives_block,
    onchain_block, etf_flows_block, treasury_block,
]


if __name__ == "__main__":
    # Offline smoke test: exercise the wired blocks against the SYNTHETIC macro
    # source (no network/key) to prove the data contract holds.
    import json
    ml = macro_liquidity_block(source="synthetic")
    fs = funding_stress_block(source="synthetic")
    print("macro_liquidity_block:", list(ml.columns), ml.shape)
    print("funding_stress_block :", list(fs.columns), fs.shape)
    print(json.dumps({
        "macro_pct_missing": {c: round(float(ml[c].isna().mean()), 3)
                              for c in ml.columns},
        "funding_pct_missing": {c: round(float(fs[c].isna().mean()), 3)
                                for c in fs.columns},
    }, indent=2))
