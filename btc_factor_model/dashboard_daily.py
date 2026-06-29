"""
dashboard.py  -- PHASE 5: the daily read
========================================

A self-contained HTML cockpit that answers, for today, five questions:

  1. CURRENT FACTOR SCORES -- where does each of the 11 parent factors sit, in
     bullish-oriented sigma? (from Phase 3 FactorEngine.parent_scores)
  2. REGIME                -- calm / elevated / stress, from trailing realized
     vol vs its own history (the same signal the regime model uses).
  3. PREDICTED RETURN      -- the latest out-of-sample model prediction for the
     chosen horizon, as a forward return. (from Phase 4 ModelEngine)
  4. ATTRIBUTION           -- today's prediction decomposed into the factor
     pushes that produced it (additive contributions).
  5. RESIDUAL              -- Σ(contributions)+bias − prediction. The proof the
     attribution actually reconciles to the number it explains (~1e-7 or less).

Open the written HTML in any browser; re-run to refresh. Everything renders from
the phase outputs, so the same call works on synthetic or live data.

Honesty notes baked into the footer: predictions are walk-forward OOS, signal
lives at the longer horizons (see the rank-IC caption), and this is research
tooling — not investment advice.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from .config import SETTINGS, HORIZONS

# ---- palette: dark macro terminal; green/red reserved for signed data -------
BG, PANEL, GRID = "#0A0E14", "#121822", "#1B2430"
TEXT, MUTED = "#C7D0DB", "#6B7785"
PRICE, ACCENT = "#F6A623", "#7AA2F7"
BULL, BEAR = "#2EE6A6", "#FF5C5C"

LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(family="ui-monospace, 'JetBrains Mono', monospace",
              color=TEXT, size=12),
    margin=dict(l=140, r=54, t=16, b=28),
    xaxis=dict(gridcolor=GRID, zerolinecolor=MUTED),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)
HORIZON_LABEL = {"fwd_1d": "1-day", "fwd_5d": "1-week", "fwd_20d": "1-month",
                 "fwd_60d": "3-month", "fwd_6m": "6-month", "fwd_12m": "12-month"}


def _div(fig) -> str:
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   config={"displayModeBar": False})


# =========================================================================== #
# regime
# =========================================================================== #
def regime_read(price: pd.Series, window: int = 20) -> dict:
    """Calm / Elevated / Stress from trailing realized vol's expanding percentile."""
    r = np.log(price.astype(float)).diff()
    rv = (r.rolling(window).std() * np.sqrt(365)).dropna()
    if rv.empty:
        return {"state": "n/a", "vol": np.nan, "pct": np.nan, "color": MUTED}
    cur = float(rv.iloc[-1])
    pct = float((rv <= cur).mean())                  # expanding-ish percentile
    if pct < 0.34:
        state, color = "CALM", BULL
    elif pct < 0.67:
        state, color = "ELEVATED", PRICE
    else:
        state, color = "STRESS", BEAR
    return {"state": state, "vol": cur, "pct": pct, "color": color,
            "asof": rv.index[-1]}


# =========================================================================== #
# indicator cards (pure HTML/CSS)
# =========================================================================== #
def _pred_card(model_results, model, horizon) -> str:
    key = (model, horizon)
    fr = model_results.fold_results.get(key)
    ic = np.nan
    if not model_results.metrics.empty:
        row = model_results.metrics.query("model==@model and horizon==@horizon")
        ic = float(row["rank_ic"].iloc[0]) if len(row) else np.nan
    if fr is None or fr.pred.empty:
        return _card("PREDICTED RETURN", "n/a", MUTED, "no OOS prediction")
    p = fr.pred.dropna()
    logret = float(p.iloc[-1])
    pctret = (np.expm1(logret)) * 100.0
    color = BULL if pctret >= 0 else BEAR
    sub = (f"{HORIZON_LABEL.get(horizon, horizon)} · {model} · "
           f"rank IC {ic:+.2f}" if ic == ic else f"{horizon} · {model}")
    asof = p.index[-1].date()
    return _card("PREDICTED RETURN", f"{pctret:+.1f}%", color,
                 sub, foot=f"as of {asof} · walk-forward OOS")


def _regime_card(reg: dict) -> str:
    if reg["state"] == "n/a":
        return _card("REGIME", "n/a", MUTED, "insufficient history")
    sub = f"realized vol {reg['vol']*100:.0f}% · {reg['pct']*100:.0f}th pctile"
    return _card("REGIME", reg["state"], reg["color"], sub,
                 foot=f"trailing 20d vol · as of {reg['asof'].date()}")


def _residual_card(model_results, model, horizon) -> str:
    d = model_results.attribution.get((model, horizon))
    if not d:
        return _card("RESIDUAL", "n/a", MUTED, "no attribution")
    res = d["reconciliation"]
    ok = abs(res) < 1e-4
    color = BULL if ok else BEAR
    mark = "reconciles ✓" if ok else "MISMATCH ✗"
    return _card("RESIDUAL", f"{res:.1e}", color, mark,
                 foot="Σ contributions + bias − prediction")


def _card(label, value, color, sub, foot="") -> str:
    foot_html = f"<div class='foot'>{foot}</div>" if foot else ""
    return f"""<div class="card">
      <div class="label">{label}</div>
      <div class="value" style="color:{color}">{value}</div>
      <div class="sub">{sub}</div>{foot_html}
    </div>"""


# =========================================================================== #
# charts
# =========================================================================== #
def _factor_scores_fig(parent_scores: pd.DataFrame) -> str:
    row = parent_scores.dropna(how="all").iloc[-1].dropna().sort_values()
    colors = [BULL if v >= 0 else BEAR for v in row.values]
    fig = go.Figure(go.Bar(
        x=row.values, y=[c.replace("_", " ") for c in row.index],
        orientation="h", marker_color=colors,
        text=[f"{v:+.2f}σ" for v in row.values],
        textposition="outside", textfont=dict(color=TEXT, size=11)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    rng = max(2.0, float(np.abs(row.values).max()) * 1.25)
    fig.update_xaxes(range=[-rng, rng], title="bullish-oriented σ  (+ tailwind / − headwind)")
    fig.update_layout(LAYOUT, height=360, showlegend=False)
    return _div(fig)


def _attribution_fig(model_results, model, horizon) -> str:
    d = model_results.attribution.get((model, horizon))
    if not d:
        return "<div class='empty'>No attribution for this selection.</div>"
    lat = d["latest"].copy()
    lat = lat.sort_values("contribution")
    colors = [BULL if v >= 0 else BEAR for v in lat["contribution"]]
    fig = go.Figure(go.Bar(
        x=lat["contribution"], y=[i.replace("_", " ") for i in lat.index],
        orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in lat["contribution"]],
        textposition="outside", textfont=dict(color=TEXT, size=10)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="contribution to predicted log-return (additive)")
    fig.update_layout(LAYOUT, height=360, showlegend=False)
    return _div(fig)


# =========================================================================== #
# page assembly
# =========================================================================== #
_STYLE = """
*{box-sizing:border-box}
body{margin:0;background:#0A0E14;color:#C7D0DB;
     font-family:ui-monospace,'JetBrains Mono',monospace}
.wrap{max-width:1080px;margin:0 auto;padding:22px}
.head{display:flex;justify-content:space-between;align-items:baseline;
      border-bottom:1px solid #1B2430;padding-bottom:12px;margin-bottom:18px}
.head h1{font-size:15px;letter-spacing:2px;margin:0;color:#C7D0DB;font-weight:600}
.head .meta{font-size:12px;color:#6B7785}
.head .meta b{color:#F6A623;font-weight:600}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}
.card{background:#121822;border:1px solid #1B2430;border-radius:10px;padding:16px 18px}
.card .label{font-size:10px;letter-spacing:1.5px;color:#6B7785;margin-bottom:8px}
.card .value{font-size:30px;font-weight:700;line-height:1.05}
.card .sub{font-size:12px;color:#C7D0DB;margin-top:6px}
.card .foot{font-size:10px;color:#6B7785;margin-top:8px}
.panel{background:#121822;border:1px solid #1B2430;border-radius:10px;
       padding:12px 14px;margin-bottom:18px}
.panel h2{font-size:11px;letter-spacing:1.5px;color:#C7D0DB;margin:2px 0 4px 4px;
          font-weight:600}
.panel .note{font-size:11px;color:#6B7785;margin:0 0 6px 4px}
.empty{color:#6B7785;padding:40px;text-align:center}
table.cone{width:100%;border-collapse:collapse;font-size:13px}
table.cone th{text-align:right;color:#6B7785;font-weight:600;font-size:10px;
  letter-spacing:1px;padding:6px 10px;border-bottom:1px solid #1B2430}
table.cone th:first-child{text-align:left}
table.cone td{text-align:right;padding:8px 10px;color:#C7D0DB;
  border-bottom:1px solid #141b25}
table.cone td:first-child{text-align:left;color:#6B7785}
.bull{color:#2EE6A6}.bear{color:#FF5C5C}
.foot-note{font-size:10px;color:#6B7785;line-height:1.6;border-top:1px solid #1B2430;
           padding-top:12px;margin-top:6px}
"""


def _cone_panel(cone) -> str:
    """3 / 6 / 12-month power-law channel: floor (low) / fair / top (high)."""
    if cone is None:
        return ""
    proj = cone.projection
    want = [("3-month", "90d"), ("6-month", "180d"), ("12-month", "1y")]
    h = cone.history
    p0 = cone.today["price"]
    low0, fair0t, high0 = (float(h["support"].iloc[-1]),
                           float(h["fair"].iloc[-1]),
                           float(h["resistance"].iloc[-1]))
    rows = [f"<tr><td>today</td><td class='bear'>${low0:,.0f}</td>"
            f"<td>${fair0t:,.0f}</td><td class='bull'>${high0:,.0f}</td>"
            f"<td>{fair0t / p0 - 1:+.0%}</td></tr>"]
    for label, key in want:
        r = proj[proj["horizon"] == key]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td class='bear'>${r['downside']:,.0f}</td>"
            f"<td>${r['fair']:,.0f}</td>"
            f"<td class='bull'>${r['upside']:,.0f}</td>"
            f"<td>{r['fair_%']:+.0%}</td></tr>")
    if not rows:
        return ""
    tdy = cone.today
    extra = ""
    if tdy.get("fair_metcalfe"):
        extra = (f" &nbsp;·&nbsp; Metcalfe fair ${tdy['fair_metcalfe']:,.0f} "
                 f"· blended ${tdy['fair_blended']:,.0f}")
    sub = (f"BTC ${tdy['price']:,.0f} · power-law fair ${tdy['fair']:,.0f} "
           f"({tdy['dev_pct']:+.0%}, {tdy['dev_pctile']:.0%}ile){extra}")
    return f"""<div class="panel">
    <h2>VALUATION CONE · 3 / 6 / 12-MONTH CHANNEL</h2>
    <p class="note">{sub}. Structural power-law walls (5th / trend / 95th pctile)
    projected forward — a plausible range, not a forecast.</p>
    <table class="cone"><thead><tr><th>Horizon</th><th>Low (floor)</th>
    <th>Fair</th><th>High (top)</th><th>Fair vs now</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table>
  </div>"""


def build_dashboard(panel: pd.DataFrame, *, factor_scores=None,
                    model_results=None, cone=None, model: str = "xgboost",
                    horizon: str = "fwd_60d",
                    out_path: str = "artifacts/dashboard.html",
                    settings=SETTINGS) -> str:
    """Render the daily cockpit. Builds Phase-3/4 outputs if not supplied."""
    if factor_scores is None:
        from .features.factor_engine import FactorEngine
        factor_scores = FactorEngine(settings).fit_transform(panel)
    if model_results is None:
        from .models.model_engine import run_model_engine
        model_results = run_model_engine(panel, models=(model,),
                                         settings=settings, horizons=[horizon])

    price = panel["price"] if "price" in panel else panel["close"]
    reg = regime_read(price)
    asof = panel.index[-1].date()
    last_px = float(price.dropna().iloc[-1])
    hz_lbl = HORIZON_LABEL.get(horizon, horizon)

    cards = (_pred_card(model_results, model, horizon)
             + _regime_card(reg)
             + _residual_card(model_results, model, horizon))
    scores_fig = _factor_scores_fig(factor_scores.parent_scores)
    attr_fig = _attribution_fig(model_results, model, horizon)
    cone_html = _cone_panel(cone)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC Factor Model · Daily Read</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_STYLE}</style></head><body><div class="wrap">

  <div class="head">
    <h1>BTC FACTOR MODEL · DAILY READ</h1>
    <div class="meta">{asof} &nbsp;·&nbsp; BTC <b>${last_px:,.0f}</b>
        &nbsp;·&nbsp; {hz_lbl} horizon &nbsp;·&nbsp; {model}</div>
  </div>

  <div class="cards">{cards}</div>

  {cone_html}

  <div class="panel">
    <h2>CURRENT FACTOR SCORES</h2>
    <p class="note">Eleven parent factors today, bullish-oriented. + = tailwind for forward BTC returns.</p>
    {scores_fig}
  </div>

  <div class="panel">
    <h2>ATTRIBUTION · TODAY'S DRIVERS</h2>
    <p class="note">The latest prediction decomposed into additive factor pushes (top contributors). These sum, with bias, to the predicted return — the RESIDUAL card above is that closure.</p>
    {attr_fig}
  </div>

  <div class="foot-note">
    Predictions are purged walk-forward, out-of-sample. Signal concentrates at the
    longer horizons (see rank IC); 1-day is near-noise. Factor scores use
    descriptive in-sample loadings for readability; the forecast path refits per
    fold. Data may be synthetic in development. Research tooling — not investment advice.
  </div>

</div></body></html>"""

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    from .data import synthetic, pipeline
    from .factors.construct import add_original_factors
    from .features.factor_engine import FactorEngine
    from .models.model_engine import run_model_engine

    syn = synthetic.generate()
    panel = pipeline.align_panel(syn["factors"], syn["price"])
    panel = add_original_factors(panel)
    panel = pipeline.add_forward_return_targets(panel)

    fs = FactorEngine().fit_transform(panel)
    mr = run_model_engine(panel, models=("xgboost",), horizons=["fwd_60d"])
    path = build_dashboard(panel, factor_scores=fs, model_results=mr,
                           model="xgboost", horizon="fwd_60d")
    print("wrote", path)
