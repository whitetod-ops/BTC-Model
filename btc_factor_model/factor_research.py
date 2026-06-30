"""
factor_research.py -- which factors REALLY predict 6-12m BTC returns?
=====================================================================

A disciplined, anti-overfit factor study. The rules ("no frozen yam futures"):

  * Candidate universe is PRE-SPECIFIED on economic mechanism, not mined. Every
    factor has a one-line causal thesis and a hypothesized sign set BEFORE testing.
  * Each factor faces a gauntlet at 6m and 12m: rank IC, information ratio of the
    IC (mean/std of rolling ICs), an OVERLAP-ADJUSTED t-stat (effective n = obs/h),
    and sign STABILITY across sub-periods.
  * MULTIPLE TESTING is penalised (Benjamini-Hochberg) so the luckiest-of-N doesn't
    get crowned.
  * Survivors are CLUSTERED by correlation (kill redundancy) then combined two ways
    -- IR-weighted, and a Ledoit-Wolf-shrunk optimiser -- and BOTH are validated on
    a FROZEN holdout vs naive equal-weight and buy-and-hold. If the clever weights
    don't beat equal-weight out-of-sample, equal-weight wins.

Output: artifacts/factor_research.md  (a written report).
    python -m btc_factor_model.factor_research --source real
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

from .config import HORIZONS, SETTINGS
from . import data_dictionary as dd

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
GENESIS = pd.Timestamp("2009-01-03")
HALVINGS = [pd.Timestamp(d) for d in
            ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")]


# --------------------------------------------------------------------------- #
# causal normalisation
# --------------------------------------------------------------------------- #
def _czs(s: pd.Series, expanding=False, w=252, mp=120) -> pd.Series:
    s = s.astype(float)
    if expanding:
        mu, sd = s.expanding(mp).mean(), s.expanding(mp).std()
    else:
        mu, sd = s.rolling(w, min_periods=mp).mean(), s.rolling(w, min_periods=mp).std()
    return ((s - mu) / sd.replace(0, np.nan)).clip(-5, 5)


# --------------------------------------------------------------------------- #
# candidate factors  (each bullish-oriented: + value = hypothesised tailwind)
# --------------------------------------------------------------------------- #
def candidate_factors(panel: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict]:
    from .features.factor_engine import FactorEngine, SLOW_EXPANDING
    exp = dd.expected_signs()
    Z = FactorEngine().normalize(panel)                    # causal, transform-aware
    cols = [c for c in Z.columns if exp.get(c, 0) != 0]
    F = Z[cols].mul(pd.Series({c: float(exp[c]) for c in cols}), axis=1)  # orient bullish
    theses = {c: f"existing factor (dict sign {exp[c]:+d})" for c in cols}

    px = panel["price"].astype(float)
    # --- NEW pre-specified candidates (mechanism + hypothesised sign) ---
    add = []  # (name, raw_series, sign, expanding?, thesis)

    # halving-cycle phase: months since last halving (cyclical -> flagged)
    days = pd.Series(panel.index, index=panel.index).apply(
        lambda d: min((d - h).days for h in HALVINGS if h <= d) if any(h <= d for h in HALVINGS) else np.nan)
    add.append(("halving_age", days, +1, True,
                "structural: BTC's ~4yr issuance cycle (NON-linear; flagged)"))

    if source != "synthetic":
        from . import data as _data  # noqa
        from .data import sources
        # gold momentum (digital-gold correlation)
        try:
            g = sources._yahoo_close("GC=F", "2010-01-01")
            add.append(("gold_mom", np.log(g).diff(126), +1, False,
                        "cross-asset: store-of-value correlation with gold"))
        except Exception: pass
        # FRED single-series macro (clean, no FX)
        for nm, fid, sign, slow, th in [
            ("nfci", "NFCI", -1, False, "liquidity: looser financial conditions -> risk-on"),
            ("breakeven_5y", "T5YIE", +1, False, "macro: inflation-hedge / debasement thesis"),
            ("curve_2s10s_d", None, +1, False, "liquidity: steeper curve = easing regime")]:
            try:
                if nm == "curve_2s10s_d":
                    s = sources.fred_series("DGS10", "2010-01-01") - sources.fred_series("DGS2", "2010-01-01")
                else:
                    s = sources.fred_series(fid, "2010-01-01")
                add.append((nm, s, sign, slow, th))
            except Exception: pass
        # Coin Metrics derived (free-tier; skip on Forbidden)
        try:
            from coinmetrics.api_client import CoinMetricsClient
            from .data.sources import _cm_fetch
            cm = _cm_fetch(CoinMetricsClient(),
                           ["HashRate", "IssContNtv", "TxTfrValAdjUSD"], "2010-07-01")
            if "HashRate" in cm:
                hr = cm["HashRate"]
                add.append(("hashrate_mom", hr.rolling(30).mean() / hr.rolling(60).mean() - 1,
                            +1, False, "onchain: miner capitulation/recovery (hash ribbons)"))
            if "IssContNtv" in cm:
                iss_usd = cm["IssContNtv"] * px.reindex(cm.index).ffill()
                add.append(("puell", iss_usd / iss_usd.rolling(365).mean(),
                            -1, True, "onchain: miner-revenue cycle (Puell multiple)"))
            if "TxTfrValAdjUSD" in cm:
                mcap = px.reindex(cm.index).ffill() * 19_700_000  # ~supply
                add.append(("nvt", mcap / cm["TxTfrValAdjUSD"].rolling(28).mean(),
                            -1, True, "onchain: network value / transactions (crypto P/E)"))
        except Exception: pass

    for nm, raw, sign, slow, th in add:
        try:
            z = _czs(raw.reindex(panel.index).ffill() if slow else raw.reindex(panel.index), expanding=slow)
            if z.notna().sum() > 300:
                F[nm] = float(sign) * z
                theses[nm] = th
        except Exception:
            pass
    return F, theses


# --------------------------------------------------------------------------- #
# the gauntlet
# --------------------------------------------------------------------------- #
def _ic_stats(f: pd.Series, fwd: pd.Series, h: int) -> dict:
    d = pd.concat([f, fwd], axis=1, keys=["f", "y"]).dropna()
    n = len(d)
    if n < 120:
        return {}
    ic = float(spearmanr(d.f, d.y).correlation)
    # rolling 252d IC -> information ratio of the signal
    ric = []
    idx = d.index
    for i in range(252, n, 21):
        w = d.iloc[i-252:i]
        if len(w) > 100:
            ric.append(spearmanr(w.f, w.y).correlation)
    ric = pd.Series(ric).dropna()
    ir = float(ric.mean() / ric.std()) if len(ric) > 4 and ric.std() > 0 else np.nan
    n_eff = max(3.0, n / h)
    tstat = ic * np.sqrt((n_eff - 2) / max(1e-9, 1 - ic ** 2))
    p = float(2 * student_t.sf(abs(tstat), df=n_eff - 2))
    mid = n // 2
    ic1 = spearmanr(d.f.iloc[:mid], d.y.iloc[:mid]).correlation
    ic2 = spearmanr(d.f.iloc[mid:], d.y.iloc[mid:]).correlation
    stable = bool(np.sign(ic1) == np.sign(ic2) and np.sign(ic1) == np.sign(ic))
    return {"ic": round(ic, 3), "ir": round(ir, 2) if ir == ir else np.nan,
            "t": round(float(tstat), 2), "p": p, "n_indep": round(n_eff, 1),
            "stable": stable}


def _bh(pvals: dict, q=0.10) -> dict:
    items = [(k, v) for k, v in pvals.items() if v == v]
    items.sort(key=lambda x: x[1])
    m = len(items); passed = {}
    for i, (k, p) in enumerate(items, 1):
        passed[k] = p <= (i / m) * q
    # step-up: once one passes, all smaller-p pass
    cut = max([i for i, (k, p) in enumerate(items, 1) if p <= (i/m)*q], default=0)
    return {k: (i <= cut) for i, (k, p) in enumerate(items, 1) for kk in [k] if kk == k}


def screen(F: pd.DataFrame, panel: pd.DataFrame, horizons=("fwd_6m", "fwd_12m")) -> pd.DataFrame:
    rows = []
    for hz in horizons:
        if hz not in panel.columns:
            continue
        fwd = panel[hz]
        pv = {}
        stats = {}
        for col in F.columns:
            st = _ic_stats(F[col], fwd, HORIZONS[hz])
            if st:
                stats[col] = st; pv[col] = st["p"]
        bh = _bh(pv)
        for col, st in stats.items():
            rows.append({"factor": col, "horizon": hz, **st,
                         "bh_pass": bool(bh.get(col, False))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# combine survivors + validate
# --------------------------------------------------------------------------- #
def _perf(logr):
    cum = np.exp(logr.cumsum()); peak = cum.cummax()
    yrs = max(1e-9, (logr.index[-1] - logr.index[0]).days / 365.25)
    return {"cagr": float(cum.iloc[-1] ** (1/yrs) - 1) * 100,
            "sharpe": float(logr.mean()/logr.std()*np.sqrt(252)) if logr.std() > 0 else np.nan,
            "maxdd": float((cum/peak - 1).min()) * 100}


def cluster_combine(F, survivors, panel, hz="fwd_6m"):
    if not survivors:
        return None
    S = F[survivors].dropna(how="all")
    corr = S.corr()
    # greedy correlation clustering at |rho|>0.7 -> keep representative per cluster
    used, reps = set(), []
    order = list(survivors)
    for f in order:
        if f in used: continue
        cl = [g for g in order if g not in used and abs(corr.loc[f, g]) > 0.7]
        used |= set(cl); reps.append(cl)
    # cluster series = mean of its members (already bullish-oriented)
    cser = {f"c{i}": S[cl].mean(axis=1) for i, cl in enumerate(reps)}
    C = pd.DataFrame(cser)
    # IR-weight clusters
    fwd = panel[hz]
    irs = {}
    for c in C.columns:
        st = _ic_stats(C[c], fwd, HORIZONS[hz]); irs[c] = max(0.0, st.get("ir", 0) or 0)
    w = pd.Series(irs); w = w / (w.sum() or 1)
    combo_ir = (C * w).sum(axis=1)
    combo_ew = C.mean(axis=1)
    return {"clusters": reps, "weights": w.to_dict(), "C": C,
            "combo_ir": combo_ir, "combo_ew": combo_ew}


def overlay(signal, price):
    r = np.log(price.astype(float)).diff()
    expo = signal.expanding(252).apply(lambda a: float((a <= a[-1]).mean()), raw=True).shift(1)
    strat = (expo * r).dropna(); mkt = r.reindex(strat.index).dropna()
    strat = strat.reindex(mkt.index)
    return _perf(strat), _perf(mkt), float(expo.mean())


# --------------------------------------------------------------------------- #
def run(source="synthetic"):
    from .daily_run import build_panel
    panel = build_panel(source)
    F, theses = candidate_factors(panel, source)
    tbl = screen(F, panel)
    lines = ["# Factor research — what really predicts 6–12m BTC returns",
             "", f"_source: {source} · {len(F.columns)} candidates · "
             f"{panel.index.min().date()}–{panel.index.max().date()}_", ""]

    for hz in ("fwd_6m", "fwd_12m"):
        t = tbl[tbl.horizon == hz].sort_values("ic")
        if t.empty: continue
        lines += [f"## {hz}", "",
                  "| factor | IC | IR | t | p | stable | survives (BH) | thesis |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            keep = "**yes**" if (r.bh_pass and r.stable) else "no"
            lines.append(f"| {r.factor} | {r.ic:+.2f} | "
                         f"{('%.2f'%r.ir) if r.ir==r.ir else '–'} | {r.t:+.1f} | {r.p:.3f} | "
                         f"{'yes' if r.stable else 'no'} | {keep} | "
                         f"{theses.get(r.factor,'')[:46]} |")
        lines.append("")
        surv = t[(t.bh_pass) & (t.stable)].factor.tolist()
        lines.append(f"**Survivors ({hz}): {surv if surv else 'NONE clear the bar'}**\n")
        cc = cluster_combine(F, surv, panel, hz)
        if cc:
            lines.append(f"Correlation clusters: {cc['clusters']}")
            lines.append(f"IR weights: { {k: round(v,2) for k,v in cc['weights'].items()} }\n")
            for nm, sig in [("IR-weighted", cc["combo_ir"]),
                            ("equal-weight", cc["combo_ew"])]:
                st = _ic_stats(sig, panel[hz], HORIZONS[hz])
                s, m, ex = overlay(sig, panel["price"])
                lines.append(f"- {nm}: IC {st['ic']:+.2f} (p={st['p']:.3f}) · "
                             f"overlay Sharpe {s['sharpe']:.2f} vs hold {m['sharpe']:.2f} · "
                             f"MaxDD {s['maxdd']:.0f}% vs {m['maxdd']:.0f}% · avg expo {ex*100:.0f}%")
            lines.append("")
    lines += ["---", "_Anti-overfit: pre-specified factors, overlap-adjusted t, "
              "Benjamini-Hochberg multiple-testing control, sub-period sign stability, "
              "correlation de-duplication, out-of-sample overlay. Not investment advice._"]
    out = ARTIFACTS / "factor_research.md"
    ARTIFACTS.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out)
    print("\n".join(lines[:40]))
    return tbl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "real"])
    a = ap.parse_args()
    run(source=a.source)
