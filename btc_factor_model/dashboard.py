"""
Daily dashboard (step 10).

Renders a self-contained, institutional 'terminal'-style HTML page from a
Results bundle: power-law valuation rail, regime state, the factor-attribution
ladder (the signature panel), a multi-horizon skill grid, and today's return
decomposition. Open the written file in any browser; refresh by re-running.

A production Streamlit version (dashboard_app.py) serves the same views live.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html

from .config import HORIZONS
from .factors.construct import power_law_fair_value
from .features import plan

# ---- palette: dark macro terminal; green/red reserved for signed data -------
BG       = "#0A0E14"
PANEL    = "#121822"
GRID     = "#1B2430"
TEXT     = "#C7D0DB"
MUTED    = "#6B7785"
PRICE    = "#F6A623"   # btc valuation rail
FAIR     = "#5B7FB5"   # power-law fair value
BULL     = "#2EE6A6"
BEAR     = "#FF5C5C"
ACCENT   = "#7AA2F7"

LAYOUT = dict(
    paper_bgcolor=PANEL, plot_bgcolor=PANEL,
    font=dict(family="ui-monospace, 'JetBrains Mono', monospace",
              color=TEXT, size=12),
    margin=dict(l=48, r=18, t=34, b=32),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)


def _fig_to_div(fig) -> str:
    return to_html(fig, include_plotlyjs=False, full_html=False,
                   config={"displayModeBar": False})


def _valuation_rail(res) -> str:
    price = res.panel["price"]
    fair = res.fair_value
    if fair is None:
        fair = power_law_fair_value(price, expanding=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price, name="BTC",
                             line=dict(color=PRICE, width=1.6)))
    fig.add_trace(go.Scatter(x=fair.index, y=fair, name="Power-law fair",
                             line=dict(color=FAIR, width=1.4, dash="dot")))
    fig.update_yaxes(type="log", title="USD (log)")
    fig.update_layout(LAYOUT, title="BTC vs power-law fair value",
                      legend=dict(orientation="h", y=1.08, x=0,
                                  bgcolor="rgba(0,0,0,0)"),
                      height=300)
    return _fig_to_div(fig)


def _deviation_panel(res) -> str:
    dev = res.panel["power_law_dev"]
    colors = np.where(dev >= 0, BEAR, BULL)  # above trend = rich (bearish fwd)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dev.index, y=dev, line=dict(color=ACCENT, width=1),
                             fill="tozeroy", fillcolor="rgba(122,162,247,0.12)",
                             name="PL deviation"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(LAYOUT, title="Power-law deviation (log, mean-reverting)",
                      height=220, showlegend=False)
    return _fig_to_div(fig)


def _attribution_ladder(res, model="xgboost", horizon="fwd_60d") -> str:
    key = (model, horizon)
    if key not in res.attribution:
        return "<div class='empty'>No attribution for selected key.</div>"
    cat = res.attribution[key]["category"].copy()
    cat = cat.sort_values("mean_contrib")
    colors = [BULL if v >= 0 else BEAR for v in cat["mean_contrib"]]
    fig = go.Figure(go.Bar(
        x=cat["mean_contrib"], y=cat.index, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in cat["mean_contrib"]],
        textposition="outside", textfont=dict(color=TEXT, size=11)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(LAYOUT, height=360,
                      title=f"Mean factor contribution to {horizon} return  ·  {model}",
                      xaxis_title="log-return contribution")
    return _fig_to_div(fig)


def _skill_grid(res, metric="rank_ic") -> str:
    piv = res.metrics.pivot(index="model", columns="horizon", values=metric)
    piv = piv.reindex(columns=list(HORIZONS))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=piv.columns, y=piv.index,
        colorscale=[[0, BEAR], [0.5, PANEL], [1, BULL]], zmid=0,
        text=np.round(piv.values, 3), texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT),
        colorbar=dict(title=metric, outlinewidth=0)))
    fig.update_layout(LAYOUT, height=240,
                      title=f"Out-of-sample skill grid · {metric} (walk-forward)")
    return _fig_to_div(fig)


def _today_decomposition(res, model="xgboost", horizon="fwd_60d") -> str:
    key = (model, horizon)
    if key not in res.attribution:
        return "<div class='empty'>No decomposition.</div>", {}
    d = res.attribution[key]["latest"].copy()
    d = d.sort_values("contribution")
    colors = [BULL if v >= 0 else BEAR for v in d["contribution"]]
    fig = go.Figure(go.Bar(
        x=d["contribution"], y=d.index, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in d["contribution"]], textposition="outside",
        textfont=dict(color=TEXT, size=10)))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    meta = res.attribution[key]["latest"].attrs
    fig.update_layout(LAYOUT, height=360,
                      title=f"Today's drivers · {horizon} · {model}")
    return _fig_to_div(fig), meta


def _originals_strip(res) -> str:
    last = res.Z.iloc[-1]
    rows = []
    labels = {
        "effective_float": "Effective Tradable Float",
        "deriv_notional_over_float": "Deriv Notional / Float",
        "treasury_company_factor": "Treasury Company Factor",
        "power_law_dev": "Power-Law Deviation",
        "funding_stress_factor": "Funding Stress (JPY carry)",
    }
    for cid, lab in labels.items():
        z = last.get(cid, np.nan)
        if pd.isna(z):
            continue
        sign = BULL if z >= 0 else BEAR
        bar = min(abs(z) / 3.0, 1.0) * 100
        side = "right" if z >= 0 else "left"
        rows.append(f"""
        <div class="orow">
          <div class="olab">{lab}</div>
          <div class="otrack">
            <div class="ofill" style="width:{bar:.0f}%;background:{sign};
                 margin-{'left' if z>=0 else 'right'}:auto;"></div>
          </div>
          <div class="oval" style="color:{sign}">{z:+.2f}σ</div>
        </div>""")
    return "\n".join(rows)


def build_dashboard(res, out_path="artifacts/dashboard.html",
                    model="xgboost", horizon="fwd_60d") -> str:
    price = res.panel["price"]
    last_px = price.dropna().iloc[-1]
    asof = price.dropna().index[-1].date()
    dev_now = res.panel["power_law_dev"].dropna().iloc[-1]

    # regime read from trailing vol percentile
    rets = np.log(price).diff()
    vol = rets.rolling(20).std()
    vol_pct = (vol.rank(pct=True)).iloc[-1]
    regime = "RISK-OFF / HIGH-VOL" if vol_pct > 0.5 else "RISK-ON / CALM"
    regime_color = BEAR if vol_pct > 0.5 else BULL

    rail = _valuation_rail(res)
    devp = _deviation_panel(res)
    ladder = _attribution_ladder(res, model, horizon)
    grid = _skill_grid(res, "rank_ic")
    today, meta = _today_decomposition(res, model, horizon)
    strip = _originals_strip(res)
    pred = meta.get("prediction", np.nan)
    pred_str = f"{pred:+.2%}" if pred == pred else "n/a"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC Factor Model · Daily</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{{--bg:{BG};--panel:{PANEL};--text:{TEXT};--muted:{MUTED};--grid:{GRID};}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--text);
    font-family:ui-monospace,'JetBrains Mono',monospace;}}
  .top{{display:flex;align-items:baseline;gap:18px;padding:16px 22px;
    border-bottom:1px solid var(--grid);flex-wrap:wrap}}
  .brand{{font-size:15px;letter-spacing:.16em;color:var(--text)}}
  .brand b{{color:{PRICE}}}
  .kpis{{display:flex;gap:26px;margin-left:auto;flex-wrap:wrap}}
  .kpi{{display:flex;flex-direction:column}}
  .kpi .l{{font-size:10px;color:var(--muted);letter-spacing:.1em}}
  .kpi .v{{font-size:16px}}
  .ribbon{{padding:7px 22px;font-size:12px;letter-spacing:.14em;
    background:{regime_color}18;border-bottom:1px solid var(--grid);
    color:{regime_color}}}
  .wrap{{display:grid;grid-template-columns:1.55fr 1fr;gap:14px;padding:14px 16px}}
  .panel{{background:var(--panel);border:1px solid var(--grid);border-radius:8px;
    padding:6px 8px}}
  .full{{grid-column:1 / -1}}
  .ohdr{{font-size:11px;color:var(--muted);letter-spacing:.14em;
    padding:10px 8px 4px}}
  .orow{{display:grid;grid-template-columns:200px 1fr 64px;align-items:center;
    gap:10px;padding:5px 8px}}
  .olab{{font-size:11px;color:var(--text)}}
  .otrack{{height:8px;background:#0c1119;border-radius:4px;overflow:hidden;
    display:flex}}
  .ofill{{height:100%;border-radius:4px}}
  .oval{{font-size:11px;text-align:right}}
  .foot{{padding:12px 22px;color:var(--muted);font-size:10.5px;
    border-top:1px solid var(--grid);line-height:1.6}}
  .empty{{padding:20px;color:var(--muted)}}
  @media(max-width:980px){{.wrap{{grid-template-columns:1fr}}}}
</style></head><body>
  <div class="top">
    <div class="brand">BTC&nbsp;<b>FACTOR&nbsp;MODEL</b>&nbsp;· daily attribution</div>
    <div class="kpis">
      <div class="kpi"><span class="l">AS OF</span><span class="v">{asof}</span></div>
      <div class="kpi"><span class="l">BTC</span><span class="v">${last_px:,.0f}</span></div>
      <div class="kpi"><span class="l">PL DEVIATION</span>
        <span class="v" style="color:{BEAR if dev_now>=0 else BULL}">{dev_now:+.2f}</span></div>
      <div class="kpi"><span class="l">MODEL VIEW {horizon}</span>
        <span class="v" style="color:{BULL if (pred==pred and pred>=0) else BEAR}">{pred_str}</span></div>
    </div>
  </div>
  <div class="ribbon">REGIME: {regime} &nbsp;·&nbsp; 20d vol pctile {vol_pct*100:.0f}
       &nbsp;·&nbsp; model: {model}</div>

  <div class="wrap">
    <div class="panel">{rail}</div>
    <div class="panel">{today}</div>
    <div class="panel full">{ladder}</div>
    <div class="panel">{devp}</div>
    <div class="panel">{grid}</div>
    <div class="panel full">
      <div class="ohdr">ORIGINAL FACTORS · CURRENT STANDARDIZED LEVEL</div>
      {strip}
    </div>
  </div>

  <div class="foot">
    Walk-forward out-of-sample. Embargo = horizon (purged overlapping targets).
    PCA refit per fold on training data only; features publication-lagged.
    Contributions are additive and reconcile to the model prediction
    (linear: β·x; XGBoost: tree-SHAP). &nbsp;|&nbsp;
    <b>Research tooling, not investment advice. Figures shown are from
    SYNTHETIC demo data unless wired to live sources.</b>
  </div>
</body></html>"""

    from pathlib import Path
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)
