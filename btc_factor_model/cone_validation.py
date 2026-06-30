"""
cone_validation.py -- does the power-law valuation cone EARN its place?
======================================================================

The factor screen (factor_research.py) asks "does the cone forecast forward
returns with a significant IC?" -- a bar nothing clears on ~16y of overlapping
data. That is the WRONG test for a valuation band. This module tests the cone's
ACTUAL claims, and tries hard to kill it:

  1. CALIBRATION   causal [5,95] channel should contain price ~90% of the time,
                   stably across eras (binomial test).
  2. ORDERING      sort days by causal cone-percentile; forward 6m/12m returns
                   should fall monotonically cheap -> rich (rank IC < 0).
  3. FROZEN HOLDOUT fit power-law on pre-2018 ONLY, freeze, test calibration +
                   ordering on 2018-2026 (truly unseen). Kills in-sample fitting.
  4. ECONOMIC      map percentile -> exposure (cheap=heavy, rich=light); compare
                   Sharpe / MaxDD / CAGR vs buy-and-hold.
  5. PLACEBO       fit the SAME cone to driftless/GBM random walks. If the cheap
                   -> rich ordering shows up there too, the edge is a mechanical
                   "revert-to-your-own-trend" artifact and we say so.
  6. BASELINE      compare vs a naive 365d log-MA deviation. If the power-law
                   doesn't beat the dumb baseline, the honest claim is just
                   "BTC mean-reverts to its long-run log trend."

Output: artifacts/cone_validation.md
    python -m btc_factor_model.cone_validation --source real
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

from .factors.construct import power_law_fair_value
GENESIS = pd.Timestamp("2009-01-03")
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
H = {"6m": 126, "12m": 252}


# --------------------------------------------------------------------------- #
# primitives (all causal)
# --------------------------------------------------------------------------- #
def deviation_causal(price: pd.Series, mp=252) -> pd.Series:
    fair = power_law_fair_value(price, expanding=True, min_periods=mp)
    return (np.log(price) - np.log(fair)).rename("dev")


def expanding_pctile(s: pd.Series, mp=252) -> pd.Series:
    return s.expanding(mp).apply(lambda a: float((a <= a[-1]).mean()), raw=True)


def fwd_logret(price: pd.Series, h: int) -> pd.Series:
    return (np.log(price).shift(-h) - np.log(price)).rename(f"fwd_{h}")


def _perf(logr: pd.Series) -> dict:
    logr = logr.dropna()
    if len(logr) < 30:
        return {"cagr": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    cum = np.exp(logr.cumsum()); peak = cum.cummax()
    yrs = max(1e-9, (logr.index[-1] - logr.index[0]).days / 365.25)
    return {"cagr": float(cum.iloc[-1] ** (1/yrs) - 1) * 100,
            "sharpe": float(logr.mean()/logr.std()*np.sqrt(252)) if logr.std() > 0 else np.nan,
            "maxdd": float((cum/peak - 1).min()) * 100}


def _rank_ic(sig: pd.Series, fwd: pd.Series) -> tuple[float, float, int]:
    d = pd.concat([sig, fwd], axis=1).dropna()
    if len(d) < 100:
        return np.nan, np.nan, len(d)
    ic = spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation
    return float(ic), round(len(d) / 252, 1), len(d)   # ic, indep-years, n


# --------------------------------------------------------------------------- #
# 1+2. calibration + ordering (causal expanding cone)
# --------------------------------------------------------------------------- #
def calibration(price: pd.Series, mp=252, window=1095) -> dict:
    # trailing-window 5/95 deviation quantiles -- mirrors the shipped cone band
    # (ValuationCone.channel_window) so this tests the walls the dashboard draws.
    dev = deviation_causal(price, mp)
    q05 = dev.rolling(window, min_periods=mp).quantile(0.05)
    q95 = dev.rolling(window, min_periods=mp).quantile(0.95)
    d = pd.concat([dev, q05, q95], axis=1).dropna()
    inside = (d.iloc[:, 0] >= d.iloc[:, 1]) & (d.iloc[:, 0] <= d.iloc[:, 2])
    n, k = len(inside), int(inside.sum())
    p = binomtest(k, n, 0.90).pvalue if n else np.nan
    return {"coverage": round(k / n, 3) if n else np.nan, "n": n, "window_d": window,
            "nominal": 0.90, "binom_p_vs_90": round(float(p), 3) if n else np.nan}


def ordering(price: pd.Series, mp=252) -> pd.DataFrame:
    dev = deviation_causal(price, mp)
    pct = expanding_pctile(dev, mp)
    rows = []
    for lbl, h in H.items():
        fwd = fwd_logret(price, h)
        ic, yrs, n = _rank_ic(pct, fwd)
        # bucket medians cheap->rich
        d = pd.concat([pct, fwd], axis=1).dropna()
        d.columns = ["pct", "f"]
        d["bk"] = pd.cut(d.pct, [0, .2, .4, .6, .8, 1.0],
                         labels=["cheap", "lo", "mid", "hi", "rich"])
        med = d.groupby("bk", observed=True).f.median()
        mono = bool(med.dropna().is_monotonic_decreasing)
        rows.append({"horizon": lbl, "rank_ic": round(ic, 3) if ic == ic else np.nan,
                     "indep_yrs": yrs, "n": n,
                     "cheap_med": round(float(med.get("cheap", np.nan)) * 100, 1),
                     "rich_med": round(float(med.get("rich", np.nan)) * 100, 1),
                     "monotone": mono})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. frozen holdout: fit pre-2018, test 2018+
# --------------------------------------------------------------------------- #
def frozen_holdout(price: pd.Series, split="2018-01-01") -> dict:
    days = pd.Series((price.index - GENESIS).days.astype(float), index=price.index)
    x, y = np.log(days), np.log(price)
    tr = price.index < pd.Timestamp(split)
    if tr.sum() < 252 or (~tr).sum() < 252:
        return {"note": "insufficient pre/post split data"}
    b, a = np.polyfit(x[tr], y[tr], 1)               # FROZEN fit, pre-2018 only
    dev = (y - (a + b * x)).rename("dev_frozen")
    sd = dev[tr].std()
    lo, hi = -1.645 * sd, 1.645 * sd                 # frozen band
    post = dev[~tr]
    inside = ((post >= lo) & (post <= hi))
    cov = float(inside.mean())
    pct_post = post.rank(pct=True)                   # rank within OOS period
    out = {"slope_b": round(float(b), 3),
           "oos_coverage": round(cov, 3), "oos_n": int((~tr).sum())}
    for lbl, h in H.items():
        ic, yrs, n = _rank_ic(pct_post, fwd_logret(price, h)[~tr])
        out[f"oos_ic_{lbl}"] = round(ic, 3) if ic == ic else np.nan
        out[f"oos_yrs_{lbl}"] = yrs
    return out


# --------------------------------------------------------------------------- #
# 4. economic backtest: valuation tilt vs HODL
# --------------------------------------------------------------------------- #
def economic(price: pd.Series, mp=252) -> dict:
    dev = deviation_causal(price, mp)
    pct = expanding_pctile(dev, mp)
    expo = (1.0 - pct).clip(0, 1).shift(1)           # cheap -> heavy, causal
    r = np.log(price).diff()
    common = pd.concat([expo, r], axis=1).dropna().index
    strat = (expo.reindex(common) * r.reindex(common))
    return {"strategy": _perf(strat), "hodl": _perf(r.reindex(common)),
            "avg_exposure": round(float(expo.reindex(common).mean()), 2)}


# --------------------------------------------------------------------------- #
# 5. placebo: same cone on random walks (is the edge an artifact?)
# --------------------------------------------------------------------------- #
def placebo(price: pd.Series, n_sims=200, seed=0, mp=252) -> dict:
    r = np.log(price).diff().dropna()
    mu, sd = r.mean(), r.std()
    rng = np.random.default_rng(seed)
    n = len(price)
    real_ic, _, _ = _rank_ic(expanding_pctile(deviation_causal(price, mp), mp),
                             fwd_logret(price, H["12m"]))
    sims = []
    for _ in range(n_sims):
        sim = pd.Series(np.exp(np.cumsum(rng.normal(mu, sd, n))) * float(price.iloc[0]),
                        index=price.index).clip(lower=1e-6)
        ic, _, _ = _rank_ic(expanding_pctile(deviation_causal(sim, mp), mp),
                            fwd_logret(sim, H["12m"]))
        if ic == ic:
            sims.append(ic)
    sims = np.array(sims)
    # one-sided: cone ordering is NEGATIVE (cheap->high). artifact if real not more
    # negative than the random-walk null.
    pctl = float((sims <= real_ic).mean()) if len(sims) else np.nan
    return {"real_ic_12m": round(real_ic, 3) if real_ic == real_ic else np.nan,
            "placebo_mean_ic": round(float(sims.mean()), 3) if len(sims) else np.nan,
            "placebo_p05": round(float(np.percentile(sims, 5)), 3) if len(sims) else np.nan,
            "real_vs_placebo_pctile": round(pctl, 3), "n_sims": len(sims)}


# --------------------------------------------------------------------------- #
# 6. baseline: naive 365d log-MA deviation
# --------------------------------------------------------------------------- #
def baseline_ma(price: pd.Series, win=365, mp=252) -> dict:
    lp = np.log(price)
    dev_ma = lp - lp.rolling(win, min_periods=win // 2).mean()
    pct = expanding_pctile(dev_ma, mp)
    out = {}
    for lbl, h in H.items():
        ic, _, _ = _rank_ic(pct, fwd_logret(price, h))
        out[f"ma_ic_{lbl}"] = round(ic, 3) if ic == ic else np.nan
    return out


# --------------------------------------------------------------------------- #
def _artifact_verdict(pctl) -> str:
    if pctl is None or pctl != pctl:
        return "inconclusive (placebo failed)"
    return ("EDGE beyond a revert-to-trend artifact" if pctl < 0.10
            else "NOT clearly beyond a revert-to-trend artifact")


def run(source="synthetic"):
    from .daily_run import build_panel
    panel = build_panel(source)
    price = panel["price"].astype(float).dropna()
    cal = calibration(price)
    order = ordering(price)
    froz = frozen_holdout(price)
    econ = economic(price)
    plac = placebo(price)
    base = baseline_ma(price)

    L = ["# Cone validation — does the power-law band earn its place?", "",
         f"_source: {source} · {price.index.min().date()}–{price.index.max().date()} "
         f"· {len(price)} obs_", ""]
    L += ["## 1. Calibration (causal [5,95] channel)",
          f"- coverage **{cal['coverage']*100:.1f}%** vs 90% nominal "
          f"(n={cal['n']}, {cal['window_d']}d trailing band, binomial p={cal['binom_p_vs_90']})", ""]
    L += ["## 2. Ordering — forward return by valuation (cheap should pay)",
          "| h | rank IC | indep yrs | cheap median | rich median | monotone |",
          "|---|---|---|---|---|---|"]
    for _, r in order.iterrows():
        L.append(f"| {r.horizon} | {r.rank_ic:+.2f} | {r.indep_yrs} | "
                 f"{r.cheap_med:+.0f}% | {r.rich_med:+.0f}% | {'yes' if r.monotone else 'no'} |")
    L += ["", "_negative IC = cheap predicts higher forward return (the claim)._", ""]
    L += ["## 3. Frozen holdout (fit pre-2018, test 2018+)"]
    if froz.get("note"):
        L += [f"- _skipped: {froz['note']} (needs real 2010+ history)_", ""]
    else:
        L += [f"- frozen slope b={froz.get('slope_b')} · OOS coverage "
              f"**{froz.get('oos_coverage','?')}** (n={froz.get('oos_n','?')})",
              f"- OOS ordering IC: 6m {froz.get('oos_ic_6m','?')} "
              f"({froz.get('oos_yrs_6m','?')}y) · 12m {froz.get('oos_ic_12m','?')} "
              f"({froz.get('oos_yrs_12m','?')}y)", ""]
    s, h = econ["strategy"], econ["hodl"]
    L += ["## 4. Economic — valuation-tilt vs buy-and-hold",
          f"- strategy: Sharpe **{s['sharpe']:.2f}**, CAGR {s['cagr']:.0f}%, "
          f"MaxDD {s['maxdd']:.0f}% (avg expo {econ['avg_exposure']})",
          f"- HODL:     Sharpe {h['sharpe']:.2f}, CAGR {h['cagr']:.0f}%, "
          f"MaxDD {h['maxdd']:.0f}%", ""]
    L += ["## 5. Placebo — same cone on random walks (artifact check)",
          f"- real 12m IC **{plac['real_ic_12m']}** vs random-walk null mean "
          f"{plac['placebo_mean_ic']} (5th pctile {plac['placebo_p05']}, "
          f"n={plac['n_sims']})",
          f"- real sits at the {plac['real_vs_placebo_pctile']} quantile of the "
          f"null → {_artifact_verdict(plac['real_vs_placebo_pctile'])}", ""]
    L += ["## 6. Baseline — vs naive 365d log-MA deviation",
          f"- power-law IC: 6m {order.iloc[0].rank_ic:+.2f}, 12m {order.iloc[1].rank_ic:+.2f}",
          f"- naive MA  IC: 6m {base['ma_ic_6m']:+.2f}, 12m {base['ma_ic_12m']:+.2f}",
          "  _(if MA matches power-law, the honest claim is 'BTC reverts to its long-run log trend')_", ""]
    L += ["---",
          "_Honest ceiling: ~15-30 independent 6-12m windows, so strict p<0.05 is "
          "unreachable. The frozen holdout + placebo are the credible evidence, not "
          "the t-stat. Not investment advice._"]
    ARTIFACTS.mkdir(exist_ok=True)
    out = ARTIFACTS / "cone_validation.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote", out)
    print("\n".join(L))
    return {"calibration": cal, "ordering": order, "frozen": froz,
            "economic": econ, "placebo": plac, "baseline": base}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"])
    run(source=ap.parse_args().source)
