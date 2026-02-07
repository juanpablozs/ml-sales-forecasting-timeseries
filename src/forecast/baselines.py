"""Simple baseline forecasters."""
from __future__ import annotations

import pandas as pd


def naive_forecast(series: pd.Series, horizon: int):
    last = series.iloc[-1]
    return [last for _ in range(horizon)]


def seasonal_naive(series: pd.Series, horizon: int, season=7):
    return [series.iloc[-season + (i % season)] for i in range(horizon)]
