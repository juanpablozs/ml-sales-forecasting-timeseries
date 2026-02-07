"""Simple baseline forecasters."""
from __future__ import annotations

import pandas as pd


def naive_forecast(series: pd.Series, horizon: int):
    """Return the last observed value repeated for `horizon` steps."""
    if len(series) == 0:
        return [0.0] * horizon
    last = float(series.iloc[-1])
    return [last for _ in range(horizon)]


def seasonal_naive(series: pd.Series, horizon: int, season: int = 7):
    """Seasonal naive using last `season` values repeated to cover horizon.

    For example, with weekly season=7, the forecast for next days uses the
    values from the last 7 days in order.
    """
    if len(series) == 0:
        return [0.0] * horizon
    if season <= 0:
        raise ValueError("season must be > 0")
    last_vals = list(map(float, series.iloc[-season:]))
    if len(last_vals) < season:
        # pad with last value if not enough history
        last_vals = ([last_vals[0]] * (season - len(last_vals))) + last_vals
    preds = [last_vals[i % season] for i in range(horizon)]
    return preds

