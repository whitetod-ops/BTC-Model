"""
macro_data.py  -- PHASE 2: macro market-data ingestion
======================================================

Pulls the macro / cross-asset block and derives the model-ready columns the data
dictionary owns in this space:

    dxy              US dollar index           (FRED DTWEXBGS  | yfinance DX-Y.NYB)
    vix              equity vol                 (FRED VIXCLS    | yfinance ^VIX)
    move_index       rates vol                  (ICE MOVE; yfinance ^MOVE / proxy)
    usdjpy           USD/JPY spot               (FRED DEXJPUS   | yfinance JPY=X)
    real_rate_10y    10y TIPS real yield        (FRED DFII10)
    hy_oas           US HY credit OAS (bps)     (FRED BAMLH0A0HYM2, ×100 from %)
    jpy_carry_diff   US2y − JP2y                (FRED DGS2 − JP 2y*)
    fed_net_liquidity  WALCL − TGA − RRP (trn)  (FRED WALCL, WTREGEN, RRPONTSYD)

Raw building blocks (walcl, wtregen, rrp_on_bn, dgs2, dgs10, dfii10, jp2y, …) are
exposed via `building_blocks()` so the net-liquidity construction is auditable.

ONE CONTRACT, SWAPPABLE SOURCE
------------------------------
    source="fred"      real FRED pull            (needs FRED_API_KEY + network)
    source="yahoo"     FX / vol via yfinance     (DXY, VIX, USDJPY, ^MOVE)
    source="synthetic" module-local offline data (NOT REAL; for unit-testing this
                       module in isolation — the full pipeline's canonical offline
                       source is data/synthetic.py)
    source="auto"      try FRED, fall back to synthetic with a loud warning

TWO THINGS THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------------
1. It does NOT apply publication lags. Series come back stamped at their FRED
   *observation* date; the pipeline's apply_publication_lags() shifts each column
   forward by data_dictionary.pub_lag_days (look-ahead channel a). Lagging here
   too would double-count.
2. It does NOT normalize. Rolling z-scores happen later in features/normalize.py.

UNIT GOTCHA (the thing that silently corrupts net liquidity)
------------------------------------------------------------
On FRED: WALCL and WTREGEN are in **millions** of USD, but RRPONTSYD is in
**billions**. `_assemble` converts RRP ×1000 before subtracting and divides the
whole thing to **trillions**. Get this wrong and net liquidity is off by ~1000×.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

# Canonical FRED series ids for the real pull.
FRED_IDS = {
    "walcl":   "WALCL",        # Fed total assets, $mn, weekly (Wed)
    "wtregen": "WTREGEN",      # Treasury General Account, $mn, weekly (Wed)
    "rrp_on_bn": "RRPONTSYD",  # Overnight RRP, $bn, daily   <-- different unit!
    "dgs2":    "DGS2",         # 2y nominal, %, daily
    "dgs10":   "DGS10",        # 10y nominal, %, daily
    "dfii10":  "DFII10",       # 10y TIPS real, %, daily
    "dxy":     "DTWEXBGS",     # broad trade-weighted USD, index, daily
    "vix":     "VIXCLS",       # VIX, index, daily
    "usdjpy":  "DEXJPUS",      # JPY per USD, daily
    "hy_oas_pct": "BAMLH0A0HYM2",  # HY OAS, **percent**, daily
}

# Columns this module is responsible for (must be dictionary ids).
MACRO_COLUMNS = [
    "dxy", "vix", "move_index", "usdjpy", "real_rate_10y",
    "hy_oas", "jpy_carry_diff", "fed_net_liquidity",
]


# =========================================================================== #
# Real adapters
# =========================================================================== #
def fred_series(series_id: str, start: str = "2017-01-01") -> pd.Series:
    """Single FRED series, indexed by observation date. Requires FRED_API_KEY."""
    if "FRED_API_KEY" not in os.environ:
        raise NotImplementedError(
            "Set FRED_API_KEY (free at fred.stlouisfed.org) and `pip install "
            "fredapi`. Series needed: " + ", ".join(sorted(set(FRED_IDS.values())))
        )
    from fredapi import Fred
    s = Fred(api_key=os.environ["FRED_API_KEY"]).get_series(
        series_id, observation_start=start)
    return pd.Series(s, name=series_id)


def _fred_raws(start: str) -> pd.DataFrame:
    """Pull every raw FRED building block into one frame (observation-dated)."""
    cols = {key: fred_series(fid, start) for key, fid in FRED_IDS.items()}
    raw = pd.DataFrame(cols)
    # MOVE and JP 2y are not free on FRED -> best-effort below, else NaN/proxy.
    raw["move"] = _try_yahoo_move(start)
    raw["jp2y"] = _jp2y_proxy(raw.index)
    # FRED HY OAS is in percent; the dictionary wants bps.
    raw["hy_oas_bps"] = raw["hy_oas_pct"] * 100.0
    return raw


def _try_yahoo_move(start: str) -> pd.Series:
    """ICE MOVE index. Try yfinance '^MOVE'; if unavailable, return NaN (the
    caller substitutes a rates-vol proxy). Replace with your vol vendor."""
    try:
        import yfinance as yf
        m = yf.download("^MOVE", start=start, interval="1d",
                        auto_adjust=False, progress=False, threads=False)
        if m is None or len(m) == 0:
            return pd.Series(dtype=float)
        if isinstance(m.columns, pd.MultiIndex):
            m.columns = m.columns.get_level_values(0)
        return m["Close"].rename("move")
    except Exception:  # noqa: BLE001
        return pd.Series(dtype=float)


def _jp2y_proxy(index: pd.Index) -> pd.Series:
    """Japan 2y yield is not free on FRED. Placeholder ~0 (JGB 2y hovered
    near/below zero through 2023). Swap in a real JP 2y feed for live use."""
    return pd.Series(0.0, index=index, name="jp2y")


def _yahoo_raws(start: str) -> pd.DataFrame:
    """FX / vol path via yfinance for shops without a FRED key."""
    import yfinance as yf
    out = {}
    tick = {"dxy": "DX-Y.NYB", "vix": "^VIX", "usdjpy": "JPY=X", "move": "^MOVE"}
    for col, t in tick.items():
        try:
            d = yf.download(t, start=start, interval="1d",
                            auto_adjust=False, progress=False, threads=False)
            if d is not None and len(d):
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                out[col] = d["Close"]
        except Exception:  # noqa: BLE001
            pass
    if not out:
        raise RuntimeError("yfinance returned nothing for the macro tickers.")
    return pd.DataFrame(out)


# =========================================================================== #
# Synthetic (module-local, offline) raws
# =========================================================================== #
def _synthetic_raws(start: str, end: str | None, seed: int = 7) -> pd.DataFrame:
    """Module-local SYNTHETIC raws — NOT REAL DATA. Plausible levels & regimes
    so this module unit-tests offline and demonstrates the real unit handling."""
    end = end or "2025-12-31"
    idx = pd.bdate_range(start, end)
    n = len(idx)
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)

    def ar1(scale, phi=0.96):
        x = np.zeros(n); e = rng.normal(0, scale, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    # Fed balance sheet (millions): logistic QE ramp to a held peak, then QT.
    qe = 1.0 / (1.0 + np.exp(-(t - 0.32) / 0.05))      # 0 -> 1 around t≈0.32
    qt = np.clip((t - 0.62) / 0.38, 0, 1)              # linear runoff after peak
    walcl = 3.9e6 + 5.0e6 * qe - 1.5e6 * qt + ar1(8e3)
    # TGA (millions): 0.3M–1.1M, choppy.
    wtregen = np.clip(5.5e5 + 3.5e5 * np.sin(2 * np.pi * t * 3) + ar1(3e4), 3e5, 1.1e6)
    # Overnight RRP (BILLIONS): ~0 -> ~2300 (2022-23) -> drains.
    rrp_on_bn = np.clip(2300 * np.exp(-((t - 0.62) ** 2) / 0.02) + ar1(40), 0, 2500)

    # Yields (%).
    dgs2 = np.clip(2.4 - 2.2 * np.exp(-((t - 0.30) ** 2) / 0.03)
                   + 4.6 * np.clip((t - 0.55) / 0.2, 0, 1) + ar1(0.05), 0.1, 5.2)
    dgs10 = np.clip(dgs2 + 0.8 - 0.9 * np.clip((t - 0.6) / 0.3, 0, 1) + ar1(0.05), 0.5, 4.9)
    dfii10 = np.clip(dgs10 - 2.3 + ar1(0.04), -1.2, 2.4)

    dxy = 95 + 12 * np.sin(2 * np.pi * t * 1.5) + 6 * np.clip((t - 0.55) / 0.2, 0, 1) + ar1(0.4)
    vix = np.clip(15 + 10 * np.abs(ar1(0.8)) + 18 * (rng.random(n) > 0.985), 9, 65)
    move = np.clip(70 + 60 * np.clip((t - 0.5) / 0.25, 0, 1) + 8 * np.abs(ar1(0.6)), 45, 170)
    usdjpy = np.clip(108 + 45 * np.clip((t - 0.5) / 0.35, 0, 1) + ar1(0.6), 100, 162)
    jp2y = np.clip(-0.15 + 0.9 * np.clip((t - 0.8) / 0.2, 0, 1) + ar1(0.02), -0.4, 1.0)
    hy_oas_bps = np.clip(330 + 500 * np.clip(ar1(0.9), 0, None) + 250 * (vix > 35), 280, 1100)

    raw = pd.DataFrame({
        "walcl": walcl, "wtregen": wtregen, "rrp_on_bn": rrp_on_bn,
        "dgs2": dgs2, "dgs10": dgs10, "dfii10": dfii10,
        "dxy": dxy, "vix": vix, "move": move, "usdjpy": usdjpy,
        "jp2y": jp2y, "hy_oas_bps": hy_oas_bps,
    }, index=idx)
    raw.attrs["synthetic"] = True
    return raw


# =========================================================================== #
# Shared derivation (used by EVERY source -> identical math)
# =========================================================================== #
def _assemble(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive the model-ready dictionary columns from raw building blocks.

    Each derived column is emitted only when its inputs are present, so a
    partial raw frame yields a partial (never crashing) output.
    """
    out = pd.DataFrame(index=raw.index)
    direct = {"dxy": "dxy", "vix": "vix", "move_index": "move",
              "usdjpy": "usdjpy", "real_rate_10y": "dfii10"}
    for dst, src in direct.items():
        if src in raw:
            out[dst] = raw[src]

    if "hy_oas_bps" in raw:
        out["hy_oas"] = raw["hy_oas_bps"]
    elif "hy_oas_pct" in raw:                       # FRED reports percent
        out["hy_oas"] = raw["hy_oas_pct"] * 100.0   # -> bps

    if {"dgs2", "jp2y"}.issubset(raw.columns):
        out["jpy_carry_diff"] = raw["dgs2"] - raw["jp2y"]

    # Fed net liquidity in USD TRILLIONS.
    #   WALCL, WTREGEN: millions.  RRPONTSYD: billions -> ×1000 to millions.
    if {"walcl", "wtregen", "rrp_on_bn"}.issubset(raw.columns):
        out["fed_net_liquidity"] = (
            raw["walcl"] - raw["wtregen"] - raw["rrp_on_bn"] * 1_000.0
        ) / 1_000_000.0
    return out


def _to_daily(df: pd.DataFrame, ffill_limit: int = 7) -> pd.DataFrame:
    """Reindex to a business-day grid and forward-fill the weekly series (WALCL,
    WTREGEN publish Wednesdays). Limit prevents stale carry across long gaps."""
    bidx = pd.bdate_range(df.index.min(), df.index.max())
    return df.reindex(bidx).ffill(limit=ffill_limit)


# =========================================================================== #
# Public entry points
# =========================================================================== #
def building_blocks(source: str = "auto", start: str = "2017-01-01",
                    end: str | None = None, *, seed: int = 7) -> pd.DataFrame:
    """Return the RAW components (daily grid) — for auditing the derivations."""
    src = source.lower()
    if src == "auto":
        try:
            raw = _fred_raws(start); src = "fred"
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"FRED pull failed ({type(e).__name__}); using SYNTHETIC "
                          f"macro data (NOT REAL).", stacklevel=2)
            raw = _synthetic_raws(start, end, seed)
    elif src == "fred":
        raw = _fred_raws(start)
    elif src == "yahoo":
        raw = _yahoo_raws(start)
    elif src == "synthetic":
        raw = _synthetic_raws(start, end, seed)
    else:
        raise ValueError(f"unknown source {source!r}")
    raw = _to_daily(raw)
    if end:
        raw = raw[raw.index <= pd.Timestamp(end)]
    raw.attrs["source"] = src
    return raw


def macro_block(source: str = "auto", start: str = "2017-01-01",
                end: str | None = None, *, columns: list[str] | None = None,
                seed: int = 7) -> pd.DataFrame:
    """Model-ready macro columns on a daily business-day grid (dictionary ids).

    Does NOT apply publication lags or normalization — those are later stages.
    """
    raw = building_blocks(source, start, end, seed=seed)
    out = _assemble(raw)
    cols = columns or [c for c in MACRO_COLUMNS if c in out.columns]
    out = out[cols]
    out.attrs["source"] = raw.attrs.get("source", source)
    return out


# =========================================================================== #
# QA
# =========================================================================== #
def qa_report(df: pd.DataFrame) -> dict:
    """Coverage, freshness, and a couple of unit/level sanity checks."""
    rep = {
        "source": df.attrs.get("source"),
        "rows": len(df),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "columns": list(df.columns),
        "pct_missing": {c: round(float(df[c].isna().mean()) * 100, 2)
                        for c in df.columns},
    }
    # cheap plausibility guards (only on columns present)
    sane = {}
    if "fed_net_liquidity" in df:
        v = df["fed_net_liquidity"].dropna()
        sane["net_liq_in_2_to_8_trn"] = bool(v.between(2, 8).mean() > 0.8)
    if "vix" in df:
        sane["vix_in_5_90"] = bool(df["vix"].dropna().between(5, 90).all())
    if "hy_oas" in df:
        sane["hy_oas_looks_like_bps"] = bool(df["hy_oas"].dropna().median() > 100)
    rep["sanity"] = sane
    return rep


if __name__ == "__main__":
    import json
    m = macro_block(source="synthetic", start="2018-01-01", end="2025-12-31")
    print(f"[{m.attrs['source']}] macro_block {m.shape}")
    print(m.tail(3).round(3).to_string())
    print(json.dumps(qa_report(m), indent=2))
