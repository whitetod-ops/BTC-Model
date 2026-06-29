"""
valuation_cone.py  -- Upside / Fair / Downside cone
===================================================

Turns "here are the factor scores" into "where are we today vs fair value, and
the plausible range from here" across 30 days to 3 years.

Three honest layers, deliberately kept separate so you can see what each is
doing:

  1. POWER-LAW CHANNEL (structural envelope).  log10(price) = a + b·log10(age).
     The fair-value line is the multi-year anchor; the channel walls are the
     historical 5th/95th percentiles of log-deviation from it. Projected forward,
     these give the structural UPSIDE (run to the top of the band) / FAIR (track
     the trend) / DOWNSIDE (fall to the bottom). This is the reliable long-horizon
     piece — it's a valuation model, not a return forecast.

  2. RETURN-DISPERSION CONE (probabilistic width).  The empirical distribution of
     h-day forward log returns, anchored at today's price. Narrow at 30 days,
     widening with horizon — the realistic "how far could it move" band. Reliable
     at short horizons; increasingly thin sample at 2y+ (flagged per row).

  3. REGIME TILT (where you sit inside it).  The current factor regime nudges the
     central path up or down within the cone. The nudge is bounded (it can never
     push the central estimate outside the historically observed range) and fades
     with horizon, because a bullish read today says a lot about 3 months out and
     very little about 3 years out.

Honesty notes: the power law is an empirical regularity, not a law — the cone
assumes it keeps holding, and the slope may be flattening. Long-horizon return
quantiles rest on very few independent samples. This is a valuation/scenario
tool, not a price prediction, and certainly not investment advice.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")
DEFAULT_HORIZONS = {"30d": 30, "90d": 90, "180d": 180,
                    "1y": 365, "2y": 730, "3y": 1095}


def _age_days(index) -> np.ndarray:
    return (pd.DatetimeIndex(index) - GENESIS).days.to_numpy().astype(float)


def causal_expanding_ols(y: pd.Series, x: pd.Series, min_periods: int = 365) -> pd.Series:
    """Predicted y from an EXPANDING (causal) OLS of y on x: at each date t the
    slope/intercept use only data up to t (running sums, O(n), no look-ahead).
    Used for both fair-value anchors -- power-law (x=log age) and Metcalfe
    (x=log active addresses)."""
    sx = x.expanding(min_periods).sum()
    sy = y.expanding(min_periods).sum()
    sxx = (x * x).expanding(min_periods).sum()
    sxy = (x * y).expanding(min_periods).sum()
    cnt = x.expanding(min_periods).count()
    b = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx).replace(0, np.nan)
    a = (sy - b * sx) / cnt
    return (a + b * x).rename("yhat")


@dataclass
class ConeResult:
    fit: dict                 # a, b, r2 of the power-law
    today: dict               # price, fair, deviation, percentile, regime, ...
    history: pd.DataFrame     # price, fair, support, resistance, deviation
    projection: pd.DataFrame  # per-horizon fair / channel / return-cone / central
    fwd_scale: float = 1.0    # forward power-law rescaled to the blended "today"


class ValuationCone:
    def __init__(self, channel_q=(0.05, 0.95), band_q=(0.10, 0.90),
                 reversion_tau_days: float = 730.0):
        self.channel_q = channel_q     # structural channel walls (deviation pctiles)
        self.band_q = band_q           # probabilistic return cone
        self.reversion_tau = reversion_tau_days   # mean-reversion speed to trend
        self._fwd_scale = 1.0          # forward power-law rescaled to blended today
        self._network = None           # active addresses (enables Metcalfe blend)
        self._metcalfe_slope = 2.0     # classic Metcalfe exponent (fixed, stable)

    # ------------------------------------------------------------------ #
    def fit(self, price: pd.Series,
            network: pd.Series | None = None) -> "ValuationCone":
        # calendar-daily grid so horizons-in-days line up with the index
        price = price.dropna().astype(float).sort_index()
        price = price.asfreq("D").ffill() if price.index.freq is None else price
        price = price.reindex(pd.date_range(price.index.min(), price.index.max(),
                                            freq="D")).ffill()
        self.price_ = price

        d = _age_days(price.index)
        x, y = np.log10(d), np.log10(price.values)
        self.b, self.a = (float(v) for v in np.polyfit(x, y, 1))   # y = a + b x
        fair_pl = 10 ** (self.a + self.b * x)                      # power-law fair
        self.fair_powerlaw_ = pd.Series(fair_pl, index=price.index)
        ss_res = ((y - (self.a + self.b * x)) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        self.r2_ = float(1 - ss_res / ss_tot)

        # The structural channel is ALWAYS the power-law (clean, smooth, and
        # holdout-validated). Metcalfe (value ~ active-addresses**2) tracks price
        # too closely to be a sensible channel, so it is computed only as a
        # SECOND valuation read shown in a corner of the chart; its cheap/rich
        # ranking is still informative (see backtest.py --anchors).
        self._network = None
        self._fwd_scale = 1.0
        self.fair_metcalfe_ = None
        self.fair_blended_today_ = None
        if network is not None and len(network.dropna()):
            addr = network.reindex(price.index).ffill().bfill()
            self._network = addr
            logn = np.log10(addr.replace(0, np.nan))
            # CAUSAL expanding intercept (matches backtest --anchors). A full-
            # sample mean was dominated by the 2010-13 era and gave an unreliable
            # present level (e.g. an absurd ~$1.4k "Metcalfe fair").
            c = (pd.Series(y, index=price.index)
                 - self._metcalfe_slope * logn).expanding(365).mean()
            fair_m = 10 ** (c + self._metcalfe_slope * logn)
            self.fair_metcalfe_ = fair_m
            self.fair_blended_today_ = float(np.sqrt(fair_pl[-1] * float(fair_m.iloc[-1])))

        self.fair_ = pd.Series(fair_pl, index=price.index)           # power-law
        self.dev_ = np.log10(price.values) - np.log10(fair_pl)
        return self

    # ------------------------------------------------------------------ #
    def fair_at(self, dates) -> np.ndarray:
        return self._fwd_scale * 10 ** (self.a + self.b * np.log10(_age_days(dates)))

    def channel_at(self, dates):
        fair = self.fair_at(dates)
        lo = np.quantile(self.dev_, self.channel_q[0])
        hi = np.quantile(self.dev_, self.channel_q[1])
        return fair * 10 ** lo, fair, fair * 10 ** hi

    def _ret_quantile(self, h: int, q: float) -> float:
        lp = np.log(self.price_.values)
        if h >= len(lp):
            return np.nan
        r = lp[h:] - lp[:-h]                      # overlapping h-day log returns
        return float(np.quantile(r, q))

    def causal_channel(self, min_periods: int = 365) -> pd.DataFrame:
        """Where the power-law channel walls would have sat IN REAL TIME at each
        past date: expanding OLS fit + expanding deviation quantiles, both using
        only data up to t (no look-ahead). At the final date this equals the
        full-sample channel, so it connects seamlessly to the forward cone."""
        price = self.price_
        x = pd.Series(np.log10(_age_days(price.index)), index=price.index)
        y = pd.Series(np.log10(price.values), index=price.index)
        sx = x.expanding(min_periods).sum()
        sy = y.expanding(min_periods).sum()
        sxx = (x * x).expanding(min_periods).sum()
        sxy = (x * y).expanding(min_periods).sum()
        cnt = x.expanding(min_periods).count()
        b = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx).replace(0, np.nan)
        a = (sy - b * sx) / cnt
        fair_c = 10 ** (a + b * x)
        dev_c = y - np.log10(fair_c)
        lo = dev_c.expanding(min_periods).quantile(self.channel_q[0])
        hi = dev_c.expanding(min_periods).quantile(self.channel_q[1])
        return pd.DataFrame({"fair_causal": fair_c.values,
                             "support_causal": (fair_c * 10 ** lo).values,
                             "resistance_causal": (fair_c * 10 ** hi).values},
                            index=price.index)

    # ------------------------------------------------------------------ #
    def project(self, horizons=DEFAULT_HORIZONS, regime_score: float = 0.0):
        today = self.price_.index[-1]
        p0 = float(self.price_.iloc[-1])
        n = len(self.price_)
        today_dev = float(self.dev_[-1])
        q_lo = np.quantile(self.dev_, self.channel_q[0])
        q_hi = np.quantile(self.dev_, self.channel_q[1])
        half = 0.5 * (q_hi - q_lo)
        # regime sets the long-run target deviation: neutral -> revert to fair (0);
        # full bull/bear -> a modest quarter-channel above/below fair. Bounded.
        target_dev = float(np.clip(regime_score * 0.5 * half, q_lo, q_hi))

        rows = []
        for name, h in horizons.items():
            date = today + pd.Timedelta(days=h)
            fair = float(self.fair_at([date])[0])
            sup, _, res = self.channel_at([date])
            sup, res = float(sup[0]), float(res[0])

            # central = mean-reversion path: start at today's deviation, drift to
            # the regime target as the horizon lengthens (sticky near-term, trend
            # long-term), then clamp inside the channel.
            revert = 1.0 - np.exp(-h / self.reversion_tau)
            central_dev = today_dev * (1 - revert) + target_dev * revert
            central = float(np.clip(fair * 10 ** central_dev, sup, res))

            # probabilistic return band (no regime) — realistic near-term width
            p_lo = p0 * np.exp(self._ret_quantile(h, self.band_q[0]))
            p_hi = p0 * np.exp(self._ret_quantile(h, self.band_q[1]))

            indep = (n - h) / h if h < n else 0      # non-overlapping windows
            band_conf = ("high" if indep >= 8 else "medium" if indep >= 3 else "low")

            def cagr(px):
                return (px / p0) ** (365.0 / h) - 1 if h >= 365 else np.nan

            rows.append({
                "horizon": name, "h_days": h, "date": date.date(),
                "downside": sup, "fair": fair, "central": central, "upside": res,
                "ret_p10": p_lo, "ret_p90": p_hi,
                "down_%": sup / p0 - 1, "fair_%": fair / p0 - 1,
                "central_%": central / p0 - 1, "up_%": res / p0 - 1,
                "down_cagr": cagr(sup), "fair_cagr": cagr(fair),
                "up_cagr": cagr(res), "band_conf": band_conf,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def run(self, price: pd.Series, regime_score: float = 0.0,
            horizons=DEFAULT_HORIZONS, network: pd.Series | None = None) -> ConeResult:
        self.fit(price, network=network)
        proj = self.project(horizons, regime_score)

        today = self.price_.index[-1]
        p0 = float(self.price_.iloc[-1])
        fair0 = float(self.fair_.iloc[-1])
        dev0 = float(self.dev_[-1])
        pctile = float((self.dev_ <= dev0).mean())

        sup, _, res = self.channel_at(self.price_.index)
        history = pd.DataFrame({"price": self.price_.values, "fair": self.fair_.values,
                                "support": sup, "resistance": res,
                                "deviation": self.dev_}, index=self.price_.index)
        history = history.join(self.causal_channel())
        blended = self._network is not None
        today_d = {"date": today.date(), "price": p0, "fair": fair0,
                   "dev_pct": p0 / fair0 - 1, "dev_pctile": pctile,
                   "regime_score": regime_score, "r2": self.r2_, "slope": self.b,
                   "blended": blended,
                   "fair_powerlaw": float(self.fair_powerlaw_.iloc[-1]),
                   "fair_metcalfe": (float(self.fair_metcalfe_.iloc[-1])
                                     if blended else None),
                   "fair_blended": self.fair_blended_today_}
        return ConeResult(fit={"a": self.a, "b": self.b, "r2": self.r2_},
                          today=today_d, history=history, projection=proj,
                          fwd_scale=self._fwd_scale)


# --------------------------------------------------------------------------- #
def regime_score_from_parents(parent_scores: pd.DataFrame, scale: float = 2.0) -> float:
    """Latest mean bullish-oriented parent score -> regime tilt in [-1, 1]."""
    row = parent_scores.dropna(how="all").iloc[-1].dropna()
    if row.empty:
        return 0.0
    return float(np.clip(row.mean() / scale, -1.0, 1.0))


# --------------------------------------------------------------------------- #
# chart (dark terminal palette, consistent with the dashboard)
# --------------------------------------------------------------------------- #
def cone_chart(res: ConeResult, out_path: str = "artifacts/valuation_cone.html",
               lookback_days: int | None = None) -> str:
    import plotly.graph_objects as go
    from plotly.io import to_html
    from pathlib import Path

    BG, PANEL, GRID, TEXT, MUTED = "#0A0E14", "#121822", "#1B2430", "#C7D0DB", "#6B7785"
    PRICE, FAIR, BULL, BEAR, ACCENT = "#F6A623", "#5B7FB5", "#2EE6A6", "#FF5C5C", "#7AA2F7"

    hist = (res.history if lookback_days is None
            else res.history.iloc[-lookback_days:])
    today = pd.Timestamp(res.today["date"])
    p0 = res.today["price"]
    fwd = pd.date_range(today, today + pd.Timedelta(days=1095), freq="W")

    # forward structural channel: grow the fair line along the power-law trend
    # (re-anchored to the blended "today" via fwd_scale), and set the walls so
    # they START at today's historical causal channel and stay parallel to the
    # trend. This keeps the cone continuous and avoids full-sample blowups (e.g.
    # the 2010-2013 era dragging a blended 5th-pctile floor to absurd lows).
    a, b = res.fit["a"], res.fit["b"]
    fair_f = res.fwd_scale * 10 ** (a + b * np.log10(_age_days(fwd)))
    _h = res.history
    _fair0 = float(_h["fair_causal"].dropna().iloc[-1])
    up_ratio = float(_h["resistance_causal"].dropna().iloc[-1] / _fair0)
    dn_ratio = float(_h["support_causal"].dropna().iloc[-1] / _fair0)
    res_f, sup_f = fair_f * up_ratio, fair_f * dn_ratio

    fig = go.Figure()
    # historical channel -- CAUSAL: where the walls would have sat in real time
    # at each past date (expanding fit + expanding deviation quantiles).
    fig.add_trace(go.Scatter(x=hist.index, y=hist["resistance_causal"],
                             line=dict(color=BULL, width=1, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["support_causal"],
                             line=dict(color=BEAR, width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(122,162,247,0.06)",
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["fair_causal"], name="Power-law fair",
                             line=dict(color=FAIR, width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["price"], name="BTC",
                             line=dict(color=PRICE, width=1.6)))
    # forward cone: structural channel
    fig.add_trace(go.Scatter(x=fwd, y=res_f, name="Upside (channel top)",
                             line=dict(color=BULL, width=1.2)))
    fig.add_trace(go.Scatter(x=fwd, y=sup_f, name="Downside (channel floor)",
                             line=dict(color=BEAR, width=1.2), fill="tonexty",
                             fillcolor="rgba(122,162,247,0.06)"))
    fig.add_trace(go.Scatter(x=fwd, y=fair_f, name="Fair (forward)",
                             line=dict(color=FAIR, width=1.2, dash="dot")))
    # central regime-tilted path through the horizon points
    proj = res.projection
    fig.add_trace(go.Scatter(
        x=[today] + [pd.Timestamp(d) for d in proj["date"]],
        y=[p0] + list(proj["central"]), name="Regime-tilted path",
        line=dict(color=ACCENT, width=2.2),
        mode="lines+markers", marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=[today], y=[p0], name="Today",
                             mode="markers", marker=dict(color=PRICE, size=9,
                             line=dict(color="#fff", width=1))))

    # vertical divider: everything left of this is realized history; right is cone
    fig.add_vline(x=today, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(x=today, yref="paper", y=1.0, showarrow=False,
                       text="today", font=dict(color=MUTED, size=10),
                       xanchor="left", yanchor="bottom")

    if res.today.get("fair_metcalfe"):
        t = res.today
        note = (f"today  ${t['price']:,.0f}<br>"
                f"power-law fair  ${t['fair_powerlaw']:,.0f}<br>"
                f"Metcalfe fair  ${t['fair_metcalfe']:,.0f}<br>"
                f"blended fair  ${t['fair_blended']:,.0f}")
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.04,
                           showarrow=False, align="left", text=note,
                           font=dict(color=TEXT, size=10),
                           bgcolor="rgba(18,24,34,0.75)", bordercolor=MUTED,
                           borderwidth=1)

    fig.update_yaxes(type="log", title="USD (log)", gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        font=dict(family="ui-monospace, monospace", color=TEXT, size=12),
        margin=dict(l=56, r=18, t=30, b=30), height=460,
        legend=dict(orientation="h", y=1.07, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10)),
        title=dict(text="BTC valuation cone — power-law channel + regime path",
                   font=dict(size=13)))
    body = to_html(fig, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": False})
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    from .data import synthetic, pipeline
    from .factors.construct import add_original_factors
    from .features.factor_engine import FactorEngine

    syn = synthetic.generate()
    panel = pipeline.align_panel(syn["factors"], syn["price"])
    panel = add_original_factors(panel)
    fs = FactorEngine().fit_transform(panel)
    rs = regime_score_from_parents(fs.parent_scores)

    cone = ValuationCone()
    res = cone.run(panel["price"], regime_score=rs)
    print(f"power-law fit: slope b={res.fit['b']:.2f}  R²={res.fit['r2']:.3f}")
    t = res.today
    print(f"today ${t['price']:,.0f} vs fair ${t['fair']:,.0f} "
          f"({t['dev_pct']:+.0%}, {t['dev_pctile']:.0%}ile)  regime {rs:+.2f}")
    cols = ["horizon", "date", "downside", "fair", "central", "upside",
            "down_%", "fair_%", "central_%", "up_%", "return_conf"]
    print(res.projection[cols].to_string(index=False))
    print("chart ->", cone_chart(res, "artifacts/valuation_cone.html"))
