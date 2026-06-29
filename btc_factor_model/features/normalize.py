"""
Normalization (step 4): rolling z-scores, look-ahead-safe.

z_t = (x_t - mu_{[t-w, t]}) / sigma_{[t-w, t]}

The window ends AT t (inclusive) and never reaches past it, so the z-score is
computable in real time. Robust mode swaps mean/std for median/MAD, which is
much steadier through BTC's fat-tailed shocks. Output is winsorized to keep a
single 2021- or 2020-style move from dominating downstream regressions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..config import NormConfig


def rolling_z(s: pd.Series, cfg: NormConfig) -> pd.Series:
    w, mp = cfg.z_window, cfg.z_min_periods
    if cfg.use_robust:
        med = s.rolling(w, min_periods=mp).median()
        mad = (s - med).abs().rolling(w, min_periods=mp).median()
        scale = 1.4826 * mad                     # MAD -> sigma equivalent
        z = (s - med) / scale.replace(0, np.nan)
    else:
        mu = s.rolling(w, min_periods=mp).mean()
        sd = s.rolling(w, min_periods=mp).std()
        z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-cfg.winsor_z, cfg.winsor_z)


def expanding_z(s: pd.Series, cfg: NormConfig) -> pd.Series:
    """Causal EXPANDING z-score (uses all history up to t, not a fixed window).
    For slow STRUCTURAL factors (power-law deviation, Mayer, MVRV, NUPL) this
    preserves mean-reversion vs the long trend; a short rolling window would turn
    them into momentum (see backtest.py --factors)."""
    mp = cfg.z_min_periods
    if cfg.use_robust:
        med = s.expanding(mp).median()
        mad = (s - med).abs().expanding(mp).median()
        z = (s - med) / (1.4826 * mad).replace(0, np.nan)
    else:
        mu = s.expanding(mp).mean()
        sd = s.expanding(mp).std()
        z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-cfg.winsor_z, cfg.winsor_z)


def normalize_frame(df: pd.DataFrame, cols: list[str],
                    cfg: NormConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = rolling_z(df[c], cfg)
    return out
