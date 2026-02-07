"""Prediction utilities for producing H-day forecasts."""
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path


def load_model(model_path: Path = Path("models/model.joblib")):
    pack = joblib.load(model_path)
    return pack["model"], pack.get("features", [])


def forecast(df: pd.DataFrame, horizon: int = 14):
    model, features = load_model()
    df_feat = df.copy()
    # naive iterative forecasting placeholder: use last value
    last_date = pd.to_datetime(df_feat["date"]).max()
    idx = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    preds = [df_feat["y"].iloc[-1]] * horizon
    return pd.DataFrame({"date": idx, "yhat": preds})
