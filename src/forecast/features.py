"""Feature engineering for time-series forecasting.

This module creates lag, rolling and calendar features ensuring no leakage:
- lag features are created via shift
- rolling stats use shifted windows (exclude current day's value)
"""
from __future__ import annotations

import pandas as pd
from typing import Iterable


def make_features(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "y",
    lags: Iterable[int] = (1, 7, 14),
    roll_windows: Iterable[int] = (7,),
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).asfreq("D")

    for lag in lags:
        df[f"lag_{lag}"] = df[value_col].shift(lag)

    for w in roll_windows:
        df[f"rolling_mean_{w}"] = df[value_col].shift(1).rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = df[value_col].shift(1).rolling(w, min_periods=1).std().fillna(0)

    # calendar
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    # Drop rows with NA in any lag (ensures features only use past data)
    feature_cols = [c for c in df.columns if c != value_col]
    df = df.dropna(subset=[f"lag_{min(lags)}"]) if lags else df

    out = df.reset_index()
    # keep only date, value and feature columns
    keep_cols = [date_col, value_col] + [c for c in out.columns if c not in [date_col, value_col]]
    return out[keep_cols]

