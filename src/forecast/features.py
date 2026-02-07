"""Feature engineering for time-series forecasting."""
from __future__ import annotations

import pandas as pd


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("D")
    df["lag_1"] = df["y"].shift(1)
    df["lag_7"] = df["y"].shift(7)
    df["lag_14"] = df["y"].shift(14)
    df["rolling_mean_7"] = df["y"].shift(1).rolling(7, min_periods=1).mean()
    df["rolling_std_7"] = df["y"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df = df.dropna()
    return df.reset_index()
