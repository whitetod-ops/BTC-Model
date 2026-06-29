"""
btc_returns.py  -- PHASE 1: BTC OHLCV ingestion + forward-return targets
========================================================================

This is the foundation of the whole model: the *target* variable. Get this
clean and look-ahead-safe before a single feature is built ("explain first,
forecast second" still needs a correct thing to explain).

Two responsibilities, nothing more:

  1. LOAD OHLCV   -- pull daily Open/High/Low/Close/Volume for BTC/USD and put
                     it on a clean, sorted, de-duplicated DatetimeIndex.
  2. FORWARD RETS -- build multi-horizon forward returns r_{t -> t+h} for every
                     horizon in config.HORIZONS, using the package convention
                         fwd_h[t] = log(Close[t+h]) - log(Close[t]).

Sources (pick one via `source=`):
  * "yfinance"  -- real pull of 'BTC-USD' (needs network + `pip install yfinance`)
  * "csv"       -- a local OHLCV file you exported from an exchange / vendor
  * "synthetic" -- a clearly-labelled offline generator (NOT real data) so the
                   pipeline runs with no network; mirrors a 7-day crypto calendar
  * "auto"      -- try yfinance, fall back to synthetic with a loud warning

LOOK-AHEAD CONTRACT
-------------------
Forward returns are *deliberately* future-looking -- that is what a target is.
They are named with the `fwd_` prefix and MUST NEVER be used as model features.
The last `h` rows of each `fwd_h` column are NaN by construction (no future bar
yet); `qa_report` verifies exactly that, which is the proof the shift points
forward and not backward. Trailing returns (`ret_1d`) use only past bars and are
safe as feature inputs.

GRID NOTE
---------
BTC trades 7 days/week, but the macro factors in the rest of the model live on a
business-day grid where the config horizons (… 252 ≈ 1 year) are *trading-day*
counts. `build_returns_panel(grid=...)` controls this:
  * grid="business" (default) -- sample Close on business days so horizon counts
    match config + the downstream macro panel (weekend moves fold into Monday).
  * grid="calendar"           -- keep all native daily bars (here 252d ≈ 8 months).
Choose "business" unless you are modelling BTC entirely on its own 7-day clock.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd

try:  # works both as a package module and standalone
    from ..config import HORIZONS
except Exception:  # pragma: no cover - standalone fallback
    HORIZONS = {"fwd_1d": 1, "fwd_5d": 5, "fwd_20d": 20,
                "fwd_60d": 60, "fwd_6m": 126, "fwd_12m": 252}

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
GENESIS = pd.Timestamp("2009-01-03")  # Bitcoin genesis block, for power-law age


# =========================================================================== #
# 1. LOADERS
# =========================================================================== #
def load_ohlcv(
    source: str = "auto",
    start: str = "2017-01-01",
    end: str | None = None,
    *,
    csv_path: str | None = None,
    ticker: str = "BTC-USD",
    seed: int = 7,
) -> pd.DataFrame:
    """Return a clean daily OHLCV frame indexed by date (columns = OHLCV_COLS).

    Always passes through `_clean_ohlcv`, so the result is guaranteed sorted,
    de-duplicated, numeric, OHLC-consistent (high>=max(o,c), low<=min(o,c)),
    and free of all-NaN close rows.
    """
    src = source.lower()
    if src == "auto":
        try:
            df = _from_yfinance(ticker, start, end)
            src = "yfinance"
        except Exception as e:  # noqa: BLE001 - any failure -> offline fallback
            warnings.warn(
                f"yfinance pull failed ({type(e).__name__}: {e}); "
                f"falling back to SYNTHETIC data. NOT REAL — dev/offline only.",
                stacklevel=2,
            )
            df = _from_synthetic(start, end, seed=seed)
    elif src == "yfinance":
        df = _from_yfinance(ticker, start, end)
    elif src == "csv":
        if not csv_path:
            raise ValueError("source='csv' requires csv_path=...")
        df = _from_csv(csv_path)
    elif src == "synthetic":
        df = _from_synthetic(start, end, seed=seed)
    else:
        raise ValueError(f"unknown source {source!r}")

    df = _clean_ohlcv(df)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    df.attrs["source"] = src
    return df


def _from_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Real pull. Requires `pip install yfinance` and outbound network."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise NotImplementedError(
            "pip install yfinance to use source='yfinance'."
        ) from e
    raw = yf.download(
        ticker, start=start, end=end, interval="1d",
        auto_adjust=False, progress=False, threads=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"yfinance returned no rows for {ticker!r}.")
    # yfinance may return a MultiIndex (field, ticker); flatten to fields.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    return raw[["open", "high", "low", "close", "volume"]]


def _from_csv(path: str) -> pd.DataFrame:
    """Load OHLCV from a local file. Flexible about column names / date col."""
    df = pd.read_csv(path)
    # find a date column
    date_col = next(
        (c for c in df.columns
         if c.lower() in ("date", "time", "timestamp", "datetime")),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.set_index(date_col)
    # map common header variants -> canonical OHLCV
    aliases = {
        "open": ["open", "o"], "high": ["high", "h"], "low": ["low", "l"],
        "close": ["close", "adj close", "adj_close", "c", "price"],
        "volume": ["volume", "vol", "v"],
    }
    lower = {c.lower(): c for c in df.columns}
    out = {}
    for canon, names in aliases.items():
        hit = next((lower[n] for n in names if n in lower), None)
        if hit is not None:
            out[canon] = pd.to_numeric(df[hit], errors="coerce")
    out = pd.DataFrame(out)
    if "close" not in out:
        raise ValueError(f"no close-like column found in {path!r}")
    for col in OHLCV_COLS:  # fill any missing OHLCV from close/zeros
        if col not in out:
            out[col] = out["close"] if col != "volume" else 0.0
    return out[OHLCV_COLS]


def _from_synthetic(start: str, end: str | None, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC OHLCV — NOT REAL DATA. Offline/dev only.

    Power-law trend (same core as data/synthetic.py: log10 P = a + b·log10(age))
    + a slow mean-reverting deviation cycle + fat-tailed daily noise, on a 7-day
    crypto calendar. OHLC is wrapped around the close with a realistic intraday
    range; volume scales with absolute return.
    """
    end = end or "2025-12-31"
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="D")          # 7-day crypto calendar
    n = len(idx)
    age = (idx - GENESIS).days.to_numpy().astype(float)

    a, b = -16.6, 5.7
    fair = 10.0 ** (a + b * np.log10(age))

    # slow AR(1) deviation from fair value + fat-tailed daily shocks
    dev = np.empty(n)
    dev[0] = 0.0
    eps = rng.normal(0, 0.05, n)
    for t in range(1, n):
        dev[t] = 0.985 * dev[t - 1] + eps[t]
    dev -= dev.mean()
    daily_noise = rng.standard_t(df=4, size=n) * 0.028
    close = fair * np.exp(dev) * np.exp(daily_noise)

    # build OHLC around close
    close_s = pd.Series(close, index=idx)
    prev_close = close_s.shift(1).fillna(close_s.iloc[0]).to_numpy()
    open_ = prev_close * np.exp(rng.normal(0, 0.004, n))        # small gap
    logret = np.abs(np.r_[0.0, np.diff(np.log(close))])
    amp = (logret + 0.012) * rng.uniform(0.6, 1.6, n)          # intraday range
    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    high = hi_base * np.exp(amp * rng.uniform(0.2, 1.0, n))
    low = lo_base * np.exp(-amp * rng.uniform(0.2, 1.0, n))
    volume = (1e9 * (close / close[0]) ** 0.5
              * np.exp(3.0 * logret) * np.exp(rng.normal(0, 0.4, n)))

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": volume},
        index=idx,
    )
    df.attrs["synthetic"] = True
    return df


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-dup, coerce numeric, enforce OHLC consistency."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    df = df[OHLCV_COLS].apply(pd.to_numeric, errors="coerce")

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna()]

    # OHLC sanity: high is the max, low is the min of the bar
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df.loc[df["volume"] < 0, "volume"] = np.nan
    return df


# =========================================================================== #
# 2. RETURNS
# =========================================================================== #
def trailing_log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Past-looking log return over `periods` bars. SAFE as a feature input."""
    lp = np.log(close.astype(float))
    return (lp - lp.shift(periods)).rename(f"ret_{periods}d")


def forward_log_returns(
    close: pd.Series, horizons: dict[str, int] = HORIZONS
) -> pd.DataFrame:
    """Targets: fwd_h[t] = log(Close[t+h]) - log(Close[t]).  fwd_ prefix = TARGET.

    The last `h` rows of column fwd_h are NaN (no t+h bar yet) -- by design.
    """
    lp = np.log(close.astype(float))
    out = {name: lp.shift(-h) - lp for name, h in horizons.items()}
    return pd.DataFrame(out, index=close.index)


def forward_simple_returns(
    close: pd.Series, horizons: dict[str, int] = HORIZONS
) -> pd.DataFrame:
    """Targets in simple (arithmetic) form: Close[t+h]/Close[t] - 1."""
    c = close.astype(float)
    out = {name: c.shift(-h) / c - 1.0 for name, h in horizons.items()}
    return pd.DataFrame(out, index=close.index)


def realized_vol(close: pd.Series, window: int = 20, annualize: int = 365) -> pd.Series:
    """Trailing realized vol of daily log returns (regime signal downstream)."""
    r = trailing_log_return(close, 1)
    return (r.rolling(window).std() * np.sqrt(annualize)).rename(f"rv_{window}d")


# =========================================================================== #
# 3. PANEL BUILDER (the Phase-1 deliverable)
# =========================================================================== #
def build_returns_panel(
    source: str = "auto",
    start: str = "2017-01-01",
    end: str | None = None,
    *,
    grid: str = "business",
    return_type: str = "log",
    horizons: dict[str, int] = HORIZONS,
    csv_path: str | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    """End-to-end Phase 1: OHLCV + trailing return + multi-horizon forward returns.

    Columns: open, high, low, close, volume, ret_1d, <one per horizon>.
    `grid="business"` reindexes Close to business days first so horizon counts
    match config (252 ≈ 1y) and the macro panel; `grid="calendar"` keeps 7-day.
    """
    ohlcv = load_ohlcv(source, start, end, csv_path=csv_path, seed=seed)

    if grid == "business":
        bidx = pd.bdate_range(ohlcv.index.min(), ohlcv.index.max())
        panel = ohlcv.reindex(bidx).ffill(limit=4)     # sample at business days
    elif grid == "calendar":
        panel = ohlcv.copy()
    else:
        raise ValueError("grid must be 'business' or 'calendar'")

    panel["ret_1d"] = trailing_log_return(panel["close"], 1)
    fwd = (forward_log_returns if return_type == "log"
           else forward_simple_returns)(panel["close"], horizons)
    panel = panel.join(fwd)

    panel.attrs["source"] = ohlcv.attrs.get("source", source)
    panel.attrs["grid"] = grid
    panel.attrs["return_type"] = return_type
    return panel


# =========================================================================== #
# 4. QA  (proves look-ahead safety + data sanity)
# =========================================================================== #
def qa_report(panel: pd.DataFrame, horizons: dict[str, int] = HORIZONS) -> dict:
    """Sanity + look-ahead checks. The fwd-NaN-tail test is the key safety proof."""
    idx = panel.index
    rep: dict = {
        "rows": len(panel),
        "start": str(idx.min().date()),
        "end": str(idx.max().date()),
        "duplicate_dates": int(idx.duplicated().sum()),
        "monotonic_index": bool(idx.is_monotonic_increasing),
        "n_nonpositive_close": int((panel["close"] <= 0).sum()),
        "ohlc_violations": int(
            ((panel["high"] < panel[["open", "close"]].max(axis=1)) |
             (panel["low"] > panel[["open", "close"]].min(axis=1))).sum()
        ),
        "max_calendar_gap_days": int(idx.to_series().diff().dt.days.max()),
        "fwd_nan_tail_ok": {},
    }
    # each fwd_h must have its LAST h rows NaN and no NaN before the tail
    for name, h in horizons.items():
        if name not in panel.columns:
            continue
        col = panel[name]
        tail_all_nan = col.iloc[-h:].isna().all()
        body_no_nan = col.iloc[:-h].notna().all()
        rep["fwd_nan_tail_ok"][name] = bool(tail_all_nan and body_no_nan)
    rep["all_fwd_safe"] = all(rep["fwd_nan_tail_ok"].values())
    return rep


if __name__ == "__main__":
    p = build_returns_panel(source="synthetic", start="2018-01-01",
                            end="2025-12-31", grid="business")
    print(f"[{p.attrs['source']}] panel {p.shape}  grid={p.attrs['grid']}")
    print(p[["close", "ret_1d", *HORIZONS]].tail(3).round(4).to_string())
    import json
    print(json.dumps(qa_report(p), indent=2))
