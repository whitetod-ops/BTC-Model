"""
backtest.py -- honest evaluation: "is the model any good?"
==========================================================

Takes the walk-forward OUT-OF-SAMPLE predictions (never in-sample) and answers
three questions without flattering the model:

  1. IS THE SIGNAL REAL?  Rank IC with an OVERLAP-ADJUSTED t-test. h-day forward
     returns sampled daily are massively overlapping, so the naive n is a lie;
     we deflate to n_independent = n / h and test significance on that. A pretty
     rank IC on 3 independent windows is noise, and this says so.

  2. IS IT TRADEABLE?  A NON-OVERLAPPING, out-of-sample strategy backtest: at
     dates spaced one horizon apart, go long when predicted return > 0 (long/flat
     by default; long/short optional), hold for the horizon, compound. Compared
     head-to-head with buy-and-hold over the same blocks: total return, CAGR,
     Sharpe, max drawdown, hit-rate, time in market.

  3. DIRECTION.  Block-level directional accuracy vs a 50% coin flip (binomial
     p-value on independent blocks).

Everything uses fold_results.pred (stitched OOS) and y_true (realized forward
log-returns). No look-ahead, no full-sample fit. Research tooling, not advice.

    python -m btc_factor_model.backtest --source real --horizon fwd_60d
    python -m btc_factor_model.backtest --source real --horizon fwd_60d --long-short
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import SETTINGS, HORIZONS
from .models.walk_forward import FoldResult

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
HZ_LABEL = {"fwd_1d": "1d", "fwd_5d": "1w", "fwd_20d": "1m", "fwd_60d": "3m",
            "fwd_6m": "6m", "fwd_12m": "12m", "fwd_24m": "2y"}


def ic_significance(fr: FoldResult, h: int) -> dict:
    """Rank IC + overlap-adjusted significance (n_independent = n / h)."""
    d = pd.concat([fr.y_true, fr.pred], axis=1, keys=["y", "p"]).dropna()
    n = len(d)
    if n < 10:
        return {"n": n, "n_indep": 0, "rank_ic": np.nan, "t": np.nan, "p": np.nan}
    ic = float(stats.spearmanr(d.y, d.p).correlation)
    n_eff = max(3.0, n / float(h))               # de-overlap
    t = ic * np.sqrt((n_eff - 2) / max(1e-9, 1 - ic ** 2))
    p = float(2 * stats.t.sf(abs(t), df=n_eff - 2))
    return {"n": n, "n_indep": round(n_eff, 1), "rank_ic": round(ic, 3),
            "t": round(float(t), 2), "p": round(p, 3)}


def _blocks(fr: FoldResult, h: int) -> pd.DataFrame:
    """Non-overlapping (pred, realized-log-return) blocks spaced h rows apart."""
    d = pd.concat([fr.pred, fr.y_true], axis=1, keys=["p", "y"]).dropna().sort_index()
    if len(d) == 0:
        return d
    return d.iloc[np.arange(0, len(d), h)]


def strategy_backtest(fr: FoldResult, h: int, long_short: bool = False) -> dict:
    """Out-of-sample long/flat (or long/short) backtest vs buy-and-hold."""
    blk = _blocks(fr, h)
    if len(blk) < 4:
        return {"error": "too few independent blocks to backtest"}
    pos = np.where(blk.p > 0, 1.0, -1.0 if long_short else 0.0)
    strat = pd.Series(pos * blk.y.values, index=blk.index)   # log returns
    mkt = blk.y                                              # buy & hold blocks
    per_yr = 252.0 / h
    yrs = max(1e-9, (blk.index[-1] - blk.index[0]).days / 365.25)

    def summ(logr: pd.Series) -> dict:
        cum = np.exp(logr.cumsum())
        peak = cum.cummax()
        return {
            "total_%": round(float(cum.iloc[-1] - 1) * 100, 1),
            "cagr_%": round(float(cum.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "sharpe": round(float(logr.mean() / logr.std() * np.sqrt(per_yr))
                            if logr.std() > 0 else np.nan, 2),
            "maxdd_%": round(float((cum / peak - 1).min()) * 100, 1),
        }

    s, m = summ(strat), summ(mkt)
    dir_hits = int((np.sign(blk.p) == np.sign(blk.y)).sum())
    nb = len(blk)
    binom = float(stats.binomtest(dir_hits, nb, 0.5).pvalue) if nb else np.nan
    return {
        "n_blocks": nb, "years": round(yrs, 1),
        "time_in_market_%": round(float((pos != 0).mean()) * 100, 0),
        "dir_acc_%": round(dir_hits / nb * 100, 1), "dir_p": round(binom, 3),
        "strategy": s, "buy_hold": m,
        "edge_cagr_%": round(s["cagr_%"] - m["cagr_%"], 1),
    }


def evaluate(panel, res, horizon: str, model: str, long_short: bool = False) -> dict:
    h = HORIZONS[horizon]
    fr = res.fold_results[(model, horizon)]
    return {"model": model, "horizon": horizon,
            "significance": ic_significance(fr, h),
            "backtest": strategy_backtest(fr, h, long_short)}


def _verdict(sig: dict, bt: dict) -> str:
    if "error" in bt:
        return "INSUFFICIENT DATA"
    sig_ok = (sig["p"] < 0.05) and (sig["rank_ic"] > 0)
    econ_ok = bt["edge_cagr_%"] > 0 and bt["strategy"]["sharpe"] >= bt["buy_hold"]["sharpe"]
    dir_ok = bt["dir_p"] < 0.10 and bt["dir_acc_%"] > 50
    score = sum([sig_ok, econ_ok, dir_ok])
    return {3: "GOOD — signal is significant, directional, and beats hold risk-adjusted",
            2: "PROMISING — passes 2 of 3 honest tests",
            1: "WEAK — only 1 of 3 tests passes; treat as marginal",
            0: "NO EDGE — fails significance, direction, and economics"}[score]


def report(ev: dict) -> str:
    sig, bt = ev["significance"], ev["backtest"]
    L = HZ_LABEL.get(ev["horizon"], ev["horizon"])
    lines = [f"\n{'='*64}",
             f" EVALUATION — {ev['model']} @ {L} horizon ({ev['horizon']})",
             f"{'='*64}",
             "\n1) IS THE SIGNAL REAL?  (overlap-adjusted)",
             f"   rank IC {sig['rank_ic']:+.3f} | {sig['n']} obs -> "
             f"{sig['n_indep']} independent | t={sig['t']} p={sig['p']}",
             f"   -> {'significant (p<0.05)' if sig['p']<0.05 and sig['rank_ic']>0 else 'NOT significant'}"]
    if "error" in bt:
        lines.append(f"\n2) TRADEABLE?  {bt['error']}")
    else:
        s, m = bt["strategy"], bt["buy_hold"]
        lines += [
            f"\n2) IS IT TRADEABLE?  ({bt['n_blocks']} non-overlapping blocks, "
            f"{bt['years']}y, {bt['time_in_market_%']:.0f}% in market)",
            f"   {'':12s}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}{'Total':>10}",
            f"   {'strategy':12s}{s['cagr_%']:>8.1f}%{s['sharpe']:>9.2f}"
            f"{s['maxdd_%']:>8.1f}%{s['total_%']:>9.1f}%",
            f"   {'buy & hold':12s}{m['cagr_%']:>8.1f}%{m['sharpe']:>9.2f}"
            f"{m['maxdd_%']:>8.1f}%{m['total_%']:>9.1f}%",
            f"   edge: {bt['edge_cagr_%']:+.1f}% CAGR vs hold",
            f"\n3) DIRECTION:  {bt['dir_acc_%']:.0f}% of blocks called right "
            f"(p={bt['dir_p']} vs coin flip)"]
    lines += [f"\nVERDICT: {_verdict(sig, bt)}",
              "(out-of-sample, non-overlapping, no look-ahead. Not investment advice.)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cone / valuation evaluation -- "were the cheap/expensive calls any good?"
# --------------------------------------------------------------------------- #
def cone_evaluation(price: pd.Series, fwd_days=(180, 365)) -> dict:
    """Test the valuation model the honest way. Using the CAUSAL power-law
    channel (expanding fit + expanding quantiles -- only past data at each date),
    bucket every day by where price sat in the channel (cheap / fair / rich) and
    look at the forward returns that ACTUALLY followed. A valuation model is
    'good' if cheap days were followed by higher returns than rich days, and if
    the channel's coverage matches its design (~90% inside the 5-95 walls)."""
    from scipy.stats import spearmanr
    from .valuation_cone import ValuationCone
    vc = ValuationCone().fit(price)
    cc = vc.causal_channel()
    px = vc.price_
    sup, res = cc["support_causal"], cc["resistance_causal"]
    mask = sup.notna()
    cover = {
        "inside_%": round(float(((px >= sup) & (px <= res))[mask].mean()) * 100, 1),
        "above_top_%": round(float((px > res)[mask].mean()) * 100, 1),
        "below_floor_%": round(float((px < sup)[mask].mean()) * 100, 1),
        "n_days": int(mask.sum()),
    }
    # causal deviation percentile: where today's deviation ranks vs its own past
    dev = pd.Series(np.log10(px.values) - np.log10(cc["fair_causal"].values),
                    index=px.index)
    pct = dev.expanding(365).apply(lambda a: float((a <= a[-1]).mean()), raw=True)
    lp = np.log(px)
    horizons = {}
    for H in fwd_days:
        fwd = lp.shift(-H) - lp                       # forward log return
        d = pd.concat([pct, fwd], axis=1, keys=["pct", "fwd"]).dropna()
        if len(d) < 60:
            continue
        ic = float(spearmanr(d.pct, d.fwd).correlation)   # expect NEGATIVE
        n_eff = max(3.0, len(d) / H)
        t = ic * np.sqrt((n_eff - 2) / max(1e-9, 1 - ic ** 2))
        from scipy import stats as _st
        p = float(2 * _st.t.sf(abs(t), df=n_eff - 2))
        cheap = d.fwd[d.pct <= 0.33]; mid = d.fwd[(d.pct > 0.33) & (d.pct < 0.67)]
        rich = d.fwd[d.pct >= 0.67]
        med = lambda x: round(float((np.exp(x.median()) - 1) * 100), 1) if len(x) else float("nan")
        horizons[H] = {
            "rank_ic": round(ic, 3), "n_indep": round(n_eff, 1), "p": round(p, 3),
            "cheap_med_%": med(cheap), "fair_med_%": med(mid), "rich_med_%": med(rich),
            "monotonic": bool(med(cheap) > med(mid) > med(rich)),
        }
    return {"coverage": cover, "horizons": horizons,
            "today_pctile": round(float(pct.dropna().iloc[-1]) * 100, 0)}


def cone_report(ev: dict) -> str:
    c = ev["coverage"]
    L = [f"\n{'='*64}", " VALUATION-MODEL EVALUATION (causal power-law channel)",
         f"{'='*64}",
         "\n1) CHANNEL CALIBRATION  (5-95 walls -> ~90% inside by design)",
         f"   inside {c['inside_%']}%  |  above top {c['above_top_%']}%  |  "
         f"below floor {c['below_floor_%']}%   ({c['n_days']} days)",
         "\n2) DID CHEAP/RICH CALLS PREDICT FORWARD RETURNS?",
         "   (median forward return by valuation bucket; cheap should beat rich)"]
    for H, h in ev["horizons"].items():
        lbl = f"{H}d (~{round(H/30)}m)"
        verdict = ("MONOTONIC + significant" if h["monotonic"] and h["p"] < 0.05
                   else "monotonic" if h["monotonic"] else "NOT monotonic")
        L += [f"   {lbl}:  cheap {h['cheap_med_%']:+.0f}%   "
              f"fair {h['fair_med_%']:+.0f}%   rich {h['rich_med_%']:+.0f}%",
              f"          rank IC {h['rank_ic']:+.3f} (p={h['p']}, "
              f"{h['n_indep']} indep) -> {verdict}"]
    L += [f"\n   today's reading: {ev['today_pctile']:.0f}th percentile of the channel "
          f"({'cheap' if ev['today_pctile']<33 else 'rich' if ev['today_pctile']>67 else 'fair'})",
          "(causal: only past data used at each date. Not investment advice.)"]
    return "\n".join(L)


def run_cone(source="synthetic"):
    from .daily_run import build_panel
    panel = build_panel(source)
    ev = cone_evaluation(panel["price"])
    print(cone_report(ev))
    return ev


# --------------------------------------------------------------------------- #
# Frozen holdout -- the strongest anti-overfit test for the valuation cone
# --------------------------------------------------------------------------- #
def cone_holdout(price: pd.Series, split: str = "2020-06-30",
                 channel_q=(0.05, 0.95), fwd_days=(180, 365)) -> dict:
    """Fit the power-law + channel on data BEFORE `split` only, FREEZE those
    parameters, then judge them on data the fit never saw. This is a true
    out-of-sample test (no expanding window quietly re-using recent data): if a
    band drawn knowing only the early years still contains later prices and still
    orders cheap/rich -> forward returns, the power law is a structural
    regularity, not a curve fit to the whole sample."""
    from scipy.stats import spearmanr
    from .valuation_cone import _age_days
    price = price.dropna().astype(float).sort_index()
    price = price.reindex(pd.date_range(price.index.min(), price.index.max(),
                                        freq="D")).ffill()
    sp = pd.Timestamp(split)
    pre, post = price[price.index < sp], price[price.index >= sp]
    if len(pre) < 365 or len(post) < 200:
        return {"error": f"need >=1y pre and >=200d post around {split}"}

    # --- fit on PRE ONLY, then freeze ---
    xp, yp = np.log10(_age_days(pre.index)), np.log10(pre.values)
    b, a = np.polyfit(xp, yp, 1)
    dev_pre = yp - (a + b * xp)
    lo, hi = np.quantile(dev_pre, channel_q[0]), np.quantile(dev_pre, channel_q[1])

    # --- apply frozen params to POST (unseen) ---
    xpo = np.log10(_age_days(post.index))
    fair = 10 ** (a + b * xpo)
    sup, res = fair * 10 ** lo, fair * 10 ** hi
    inside = float(((post.values >= sup) & (post.values <= res)).mean())
    above = float((post.values > res).mean())
    below = float((post.values < sup).mean())

    # cheap/rich on POST, ranked against the FROZEN pre-split deviation distribution
    dev_po = np.log10(post.values) - (a + b * xpo)
    pct = np.array([float((dev_pre <= d).mean()) for d in dev_po])
    lp = np.log(pd.Series(post.values, index=post.index))
    horizons = {}
    for H in fwd_days:
        fwd = (lp.shift(-H) - lp)
        d = pd.DataFrame({"pct": pct, "fwd": fwd.values}, index=post.index).dropna()
        if len(d) < 60:
            continue
        ic = float(spearmanr(d.pct, d.fwd).correlation)
        med = lambda x: round(float((np.exp(x.median()) - 1) * 100), 1) if len(x) else float("nan")
        cheap, mid, rich = d.fwd[d.pct <= .33], d.fwd[(d.pct > .33) & (d.pct < .67)], d.fwd[d.pct >= .67]
        horizons[H] = {"rank_ic": round(ic, 3), "cheap_%": med(cheap),
                       "fair_%": med(mid), "rich_%": med(rich),
                       "monotonic": bool(med(cheap) > med(mid) > med(rich)),
                       "n": len(d)}
    return {"split": split, "pre_years": round(len(pre)/365.25, 1),
            "post_years": round(len(post)/365.25, 1), "slope_b": round(float(b), 2),
            "inside_%": round(inside*100, 1), "above_%": round(above*100, 1),
            "below_%": round(below*100, 1), "horizons": horizons}


def cone_holdout_report(h: dict) -> str:
    if "error" in h:
        return f"\nFrozen holdout: {h['error']}"
    L = [f"\n{'='*64}",
         f" FROZEN HOLDOUT -- fit on pre-{h['split']} only, test on the rest",
         f"{'='*64}",
         f" fit on {h['pre_years']}y (slope b={h['slope_b']}), tested on "
         f"{h['post_years']}y the fit NEVER saw",
         f"\n COVERAGE on unseen data (target ~90% inside): "
         f"{h['inside_%']}% inside | {h['above_%']}% above | {h['below_%']}% below",
         "\n CHEAP/RICH -> forward return, on unseen data:"]
    for H, x in h["horizons"].items():
        L.append(f"   {H}d (~{round(H/30)}m):  cheap {x['cheap_%']:+.0f}%  "
                 f"fair {x['fair_%']:+.0f}%  rich {x['rich_%']:+.0f}%   "
                 f"(IC {x['rank_ic']:+.2f}, {'ordered' if x['monotonic'] else 'NOT ordered'})")
    ok = h["inside_%"] >= 60 and all(x["monotonic"] for x in h["horizons"].values())
    L.append(f"\n VERDICT: {'power law held out-of-sample -- structural, not curve-fit' if ok else 'broke down out-of-sample -- treat power law with caution'}")
    L.append("(true holdout: parameters frozen on early data only. Not investment advice.)")
    return "\n".join(L)


def run_cone_holdout(source="synthetic", split="2020-06-30"):
    from .daily_run import build_panel
    h = cone_holdout(build_panel(source)["price"], split=split)
    print(cone_holdout_report(h))
    return h


# --------------------------------------------------------------------------- #
# Multi-anchor valuation -- power-law vs Metcalfe vs blend (does triangulation
# help, tested on the same honest cheap/rich -> forward-return bar?)
# --------------------------------------------------------------------------- #
def _fair_powerlaw(price: pd.Series) -> pd.Series:
    from .valuation_cone import _age_days, causal_expanding_ols
    x = pd.Series(np.log10(_age_days(price.index)), index=price.index)
    y = pd.Series(np.log10(price.values), index=price.index)
    return pd.Series(10 ** causal_expanding_ols(y, x).values, index=price.index)


def _fair_metcalfe(price: pd.Series, addr: pd.Series, slope: float = 2.0) -> pd.Series:
    """Classic Metcalfe: value proportional to active-addresses**2 (slope FIXED
    at 2, not fit -- a free slope is unstable because BTC price grew ~5 orders of
    magnitude while addresses grew ~1, inferring an absurd exponent that
    overflows). Only the level (intercept) is fit, causally (expanding mean of
    log price - 2*log addresses). One parameter, can't explode."""
    addr = addr.reindex(price.index).ffill()
    logp = np.log(price.astype(float))
    logn = np.log(addr.replace(0, np.nan).astype(float))
    intercept = (logp - slope * logn).expanding(365).mean()   # causal level
    fair = np.exp((intercept + slope * logn).clip(upper=40))  # clip guards overflow
    return pd.Series(fair.values, index=price.index)


def valuation_signal_eval(price: pd.Series, fair: pd.Series, fwd_days=(180, 365)) -> dict:
    """Cheap/rich (vs a causal fair value) -> forward returns. Anchor-agnostic."""
    from scipy.stats import spearmanr
    from scipy import stats as _st
    dev = (np.log(price) - np.log(fair)).dropna()
    pct = dev.expanding(365).apply(lambda a: float((a <= a[-1]).mean()), raw=True)
    lp = np.log(price)
    out = {}
    for H in fwd_days:
        d = pd.concat([pct, lp.shift(-H) - lp], axis=1, keys=["pct", "fwd"]).dropna()
        if len(d) < 60:
            continue
        ic = float(spearmanr(d.pct, d.fwd).correlation)
        n_eff = max(3.0, len(d) / H)
        p = float(2 * _st.t.sf(abs(ic * np.sqrt((n_eff - 2) / max(1e-9, 1 - ic ** 2))),
                               df=n_eff - 2))
        med = lambda x: round(float((np.exp(x.median()) - 1) * 100), 1) if len(x) else float("nan")
        cheap, mid, rich = d.fwd[d.pct <= .33], d.fwd[(d.pct > .33) & (d.pct < .67)], d.fwd[d.pct >= .67]
        out[H] = {"ic": round(ic, 3), "p": round(p, 3),
                  "cheap": med(cheap), "fair": med(mid), "rich": med(rich),
                  "spread": round(med(cheap) - med(rich), 1)}
    return out


def run_anchors(source="synthetic"):
    from .daily_run import build_panel
    from .valuation_cone import ValuationCone
    panel = build_panel(source)
    if "active_addresses" not in panel.columns:
        print("No active_addresses in panel -> Metcalfe needs Coin Metrics AdrActCnt.")
        return
    vc = ValuationCone().fit(panel["price"]); px = vc.price_
    addr = panel["active_addresses"].reindex(px.index).ffill()
    pl = _fair_powerlaw(px); me = _fair_metcalfe(px, addr)
    bl = pd.Series(np.sqrt(pl.values * me.values), index=px.index)   # geomean blend

    print(f"\n{'='*70}\n MULTI-ANCHOR VALUATION  (cheap/rich -> forward return; spread=cheap-rich)"
          f"\n{'='*70}")
    for name, fair in [("power-law", pl), ("metcalfe ", me), ("blended  ", bl)]:
        ev = valuation_signal_eval(px, fair)
        for H, x in ev.items():
            print(f" {name} {H}d: cheap {x['cheap']:+6.0f}%  fair {x['fair']:+6.0f}%  "
                  f"rich {x['rich']:+6.0f}%  | spread {x['spread']:+6.0f}  "
                  f"IC {x['ic']:+.2f} (p={x['p']})")
        print()
    # how much do the two anchors disagree today? (divergence = its own signal)
    div = float(np.log(me.iloc[-1]) - np.log(pl.iloc[-1]))
    print(f" today: power-law fair ${pl.iloc[-1]:,.0f} | metcalfe fair ${me.iloc[-1]:,.0f} "
          f"| blended ${bl.iloc[-1]:,.0f}")
    print(f" anchor divergence: {div:+.0%}  "
          f"({'Metcalfe richer (caution)' if div>0.15 else 'Metcalfe cheaper' if div<-0.15 else 'anchors agree'})")
    print("(higher spread / more negative IC = stronger valuation signal. Not advice.)")


# --------------------------------------------------------------------------- #
# Factor stability by volatility regime (descriptive diagnostic, not a tunable)
# --------------------------------------------------------------------------- #
def run_regime(source="synthetic", horizons=("fwd_60d", "fwd_6m")):
    """For each parent factor, its rank IC vs forward returns WITHIN each
    volatility regime (calm / elevated / stress). Exploratory: per-regime
    samples are small, so read these as suggestive, not significant."""
    from scipy.stats import spearmanr
    from .daily_run import build_panel
    from .features.factor_engine import FactorEngine

    panel = build_panel(source)
    fs = FactorEngine().fit_transform(panel)
    scores = fs.parent_scores
    price = panel["price"].astype(float)

    rv = np.log(price).diff().rolling(20).std() * np.sqrt(252)
    pct = rv.expanding(252).apply(lambda a: float((a <= a[-1]).mean()), raw=True)
    regime = pd.Series(np.where(pct < 0.34, "calm",
                       np.where(pct < 0.67, "elevated", "stress")),
                       index=price.index)

    def ic(a, b):
        d = pd.concat([a, b], axis=1).dropna()
        return spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation if len(d) >= 30 else np.nan

    bar = "=" * 72
    print("\n" + bar)
    print(" FACTOR STABILITY BY VOLATILITY REGIME  (rank IC vs forward return)")
    print(bar)
    for hz in horizons:
        if hz not in panel.columns:
            continue
        fwd = panel[hz]
        print("\n %s:   %-22s%9s%10s%9s%9s" %
              (hz, "factor", "calm", "elevated", "stress", "all"))
        for fac in scores.columns:
            f_ = scores[fac]
            vals = [ic(f_[regime == reg], fwd[regime == reg])
                    for reg in ("calm", "elevated", "stress")]
            vals.append(ic(f_, fwd))
            cells = "".join(("%9.2f" % v) if v == v else ("%9s" % "-") for v in vals)
            print(" %6s%-22s%s" % ("", fac.replace("_", " "), cells))
    n = regime.value_counts()
    print("\n regime days: calm %d  elevated %d  stress %d  (small cells -> suggestive only)"
          % (n.get("calm", 0), n.get("elevated", 0), n.get("stress", 0)))
    print("(bullish-oriented factors: + IC = predicts in the expected direction. Not advice.)")


# --------------------------------------------------------------------------- #
# Factor sign-check: assumed economic sign vs ACTUAL realized rank IC
# --------------------------------------------------------------------------- #
def run_factors(source="synthetic", horizons=("fwd_60d", "fwd_6m")):
    """For every factor, compare its dictionary expected_sign to the realized
    rank IC (causal normalized feature vs forward return). 'OK' means the real
    relationship points the way the economic prior assumes. Disagreements tell
    you a prior may not hold for BTC -- NOT a licence to flip signs (that would
    be overfitting); just a flag for a closer look."""
    from scipy.stats import spearmanr
    from .daily_run import build_panel
    from .features.factor_engine import FactorEngine
    from . import data_dictionary as dd

    panel = build_panel(source)
    Z = FactorEngine().normalize(panel)
    exp = dd.expected_signs()
    feats = [v.id for v in dd.VARIABLES
             if v.id in Z.columns and exp.get(v.id, 0) != 0]

    def ic(f, hz):
        d = pd.concat([Z[f], panel[hz]], axis=1).dropna()
        return spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation if len(d) >= 50 else float("nan")

    print("\n" + "=" * 70)
    print(" FACTOR SIGN-CHECK  (expected economic sign vs realized rank IC)")
    print("=" * 70)
    print(" %-20s%6s%10s%10s%8s" % ("factor", "exp", "IC 60d", "IC 6m", "6m OK?"))
    for f in feats:
        i60, i6 = ic(f, "fwd_60d"), ic(f, "fwd_6m")
        ok = "-"
        if i6 == i6:
            ok = "yes" if (i6 > 0) == (exp[f] > 0) else "NO"
        s60 = "%10.2f" % i60 if i60 == i60 else "%10s" % "-"
        s6 = "%10.2f" % i6 if i6 == i6 else "%10s" % "-"
        print(" %-20s%6d%s%s%8s" % (f.replace("_", " "), exp[f], s60, s6, ok))
    print("\n (+exp expects +IC; -exp expects -IC. 'NO' = real data disagrees with the prior.)")
    print(" (descriptive only -- do not flip signs to fit the sample. Not advice.)")


# --------------------------------------------------------------------------- #
# Experiment: fixed-weight CONDITIONS COMPOSITE (parameter-free, causal)
# --------------------------------------------------------------------------- #
def conditions_composite(panel) -> pd.Series:
    """Equal-weight, bullish-oriented average of every factor's causal z-score.
    Signs come from economic priors (data_dictionary.expected_sign), NOT from
    performance, and weights are all equal -- so there is nothing fit to returns
    and nothing to overfit. This is the honest 'can the factors time anything?'
    test: a transparent composite, not a black-box model."""
    from .features.factor_engine import FactorEngine
    from . import data_dictionary as dd
    Z = FactorEngine().normalize(panel)
    exp = dd.expected_signs()
    feats = [c for c in Z.columns if exp.get(c, 0) != 0]
    signs = pd.Series({c: float(exp[c]) for c in feats})
    oriented = Z[feats].mul(signs, axis=1)
    return oriented.mean(axis=1, skipna=True).rename("composite")


def _perf(logr):
    cum = np.exp(logr.cumsum())
    yrs = max(1e-9, (logr.index[-1] - logr.index[0]).days / 365.25)
    peak = cum.cummax()
    return {"cagr_%": round(float(cum.iloc[-1] ** (1 / yrs) - 1) * 100, 1),
            "sharpe": round(float(logr.mean() / logr.std() * np.sqrt(252))
                            if logr.std() > 0 else float("nan"), 2),
            "maxdd_%": round(float((cum / peak - 1).min()) * 100, 1),
            "total_%": round(float(cum.iloc[-1] - 1) * 100, 1)}


def run_composite(source="synthetic", horizons=("fwd_60d", "fwd_6m", "fwd_12m")):
    from scipy.stats import spearmanr
    from scipy import stats as st
    from .daily_run import build_panel
    from .config import HORIZONS

    panel = build_panel(source)
    comp = conditions_composite(panel)
    price = panel["price"].astype(float)

    ml = {}
    mlcsv = ARTIFACTS / "skill_metrics.csv"
    if mlcsv.exists():
        m = pd.read_csv(mlcsv)
        m = m[m.model == "xgboost"]
        ml = dict(zip(m.horizon, m.rank_ic))

    print("\n" + "=" * 72)
    print(" FIXED-WEIGHT CONDITIONS COMPOSITE vs ML  (rank IC, overlap-adjusted)")
    print("=" * 72)
    print(" %-9s%10s%9s%9s%14s" % ("horizon", "comp IC", "n_ind", "p", "xgb IC (last)"))
    for hz in horizons:
        if hz not in panel.columns:
            continue
        h = HORIZONS[hz]
        d = pd.concat([comp, panel[hz]], axis=1).dropna()
        if len(d) < 60:
            continue
        ic = float(spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation)
        n_eff = max(3.0, len(d) / h)
        p = float(2 * st.t.sf(abs(ic * np.sqrt((n_eff - 2) / max(1e-9, 1 - ic ** 2))),
                              df=n_eff - 2))
        mlref = ("%+.2f" % ml[hz]) if hz in ml else "n/a"
        print(" %-9s%+10.3f%9.1f%9.3f%14s" % (hz, ic, n_eff, p, mlref))

    # half-sample stability (parameter-free, so this is a fair robustness read)
    hz = "fwd_60d"
    if hz in panel.columns:
        d = pd.concat([comp, panel[hz]], axis=1).dropna()
        mid = len(d) // 2
        ic1 = spearmanr(d.iloc[:mid, 0], d.iloc[:mid, 1]).correlation
        ic2 = spearmanr(d.iloc[mid:, 0], d.iloc[mid:, 1]).correlation
        print("\n stability (fwd_60d): first-half IC %+.2f | second-half IC %+.2f"
              % (ic1, ic2))

    # risk-overlay backtest: exposure = expanding percentile of the composite
    r = np.log(price).diff()
    expo = comp.expanding(252).apply(lambda a: float((a <= a[-1]).mean()), raw=True).shift(1)
    strat = (expo * r).dropna()
    mkt = r.reindex(strat.index).dropna()
    strat = strat.reindex(mkt.index)
    s, m = _perf(strat), _perf(mkt)
    print("\n RISK OVERLAY (scale BTC exposure 0-1 by composite percentile, vs hold):")
    print("   %-12s%9s%9s%9s%10s" % ("", "CAGR", "Sharpe", "MaxDD", "Total"))
    print("   %-12s%8.1f%%%9.2f%8.1f%%%9.1f%%" %
          ("overlay", s["cagr_%"], s["sharpe"], s["maxdd_%"], s["total_%"]))
    print("   %-12s%8.1f%%%9.2f%8.1f%%%9.1f%%" %
          ("buy & hold", m["cagr_%"], m["sharpe"], m["maxdd_%"], m["total_%"]))
    print("   avg exposure %.0f%%  |  Sharpe edge %+.2f  |  drawdown change %+.1fpp"
          % (float(expo.mean()) * 100, s["sharpe"] - m["sharpe"], s["maxdd_%"] - m["maxdd_%"]))
    print("(parameter-free composite. + Sharpe edge / shallower MaxDD = useful as a risk dial.)")
    print("(not advice.)")


def run(source="synthetic", horizon="fwd_60d", models=("elastic_net", "xgboost"),
        long_short=False):
    from .daily_run import build_panel
    from .models.model_engine import run_model_engine
    panel = build_panel(source)
    res = run_model_engine(panel, models=tuple(models), horizons=[horizon])
    print(f"\nBacktest source={source}  horizon={horizon}  "
          f"{'long/short' if long_short else 'long/flat'}")
    for mdl in models:
        ev = evaluate(panel, res, horizon, mdl, long_short)
        print(report(ev))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Honest OOS evaluation of the model.")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--horizon", default="fwd_60d", choices=list(HORIZONS))
    ap.add_argument("--models", nargs="+", default=["elastic_net", "xgboost"])
    ap.add_argument("--long-short", action="store_true",
                    help="allow shorting when predicted return < 0 (default long/flat)")
    ap.add_argument("--cone", action="store_true",
                    help="evaluate the valuation cone (cheap/rich -> forward returns) "
                         "instead of the signal backtest")
    ap.add_argument("--cone-holdout", action="store_true",
                    help="frozen-holdout test: fit power law on early data only, "
                         "test on unseen later data")
    ap.add_argument("--anchors", action="store_true",
                    help="compare power-law vs Metcalfe vs blended fair value")
    ap.add_argument("--regime", action="store_true",
                    help="factor rank IC split by volatility regime (calm/elevated/stress)")
    ap.add_argument("--factors", action="store_true",
                    help="factor sign-check: expected economic sign vs realized rank IC")
    ap.add_argument("--composite", action="store_true",
                    help="fixed-weight conditions composite vs ML + risk-overlay backtest")
    ap.add_argument("--split", default="2020-06-30",
                    help="holdout split date (YYYY-MM-DD)")
    a = ap.parse_args()
    if a.composite:
        run_composite(source=a.source)
    elif a.factors:
        run_factors(source=a.source)
    elif a.regime:
        run_regime(source=a.source)
    elif a.anchors:
        run_anchors(source=a.source)
    elif a.cone_holdout:
        run_cone_holdout(source=a.source, split=a.split)
    elif a.cone:
        run_cone(source=a.source)
    else:
        run(source=a.source, horizon=a.horizon, models=tuple(a.models),
            long_short=a.long_short)
