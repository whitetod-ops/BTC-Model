"""
Base, look-ahead-safe transforms applied to raw series before normalization.

Every function here is causal: the value at row t uses only rows <= t. No
centering on the full sample, no two-sided rolling windows.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def log_diff(s: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(s).diff(periods)


def pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.pct_change(periods)


def yoy_pct_change(s: pd.Series, periods: int = 252) -> pd.Series:
    return s.pct_change(periods)


def diff(s: pd.Series, periods: int = 1) -> pd.Series:
    return s.diff(periods)


def level(s: pd.Series) -> pd.Series:
    return s


def exp_decay_events(events: pd.Series, halflife: float = 30.0) -> pd.Series:
    """Decay sparse signed events to a daily series (causal)."""
    return events.ewm(halflife=halflife, adjust=False).mean()


# dispatch table referenced by the dictionary's `transform` field
TRANSFORMS = {
    "log_diff": log_diff,
    "pct_change": pct_change,
    "yoy_pct_change": yoy_pct_change,
    "diff": diff,
    "level": level,
    "none": level,
    "level_then_diff": lambda s: s.diff(),
    "decay": exp_decay_events,
}


def apply_transform(s: pd.Series, name: str) -> pd.Series:
    fn = TRANSFORMS.get(name, level)
    return fn(s)
