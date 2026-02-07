"""Prediction utilities for producing H-day forecasts.

This implements iterative forecasting when features use lags/rolling stats:
- load trained model and feature list
- iteratively predict day t+1, append predicted y, recompute features and continue
"""
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path


def load_model(model_path: Path = Path("models/model.joblib")):
    pack = joblib.load(model_path)
    return pack["model"], pack.get("metadata", {}), pack.get("metadata", {}).get("features", [])


def forecast(df: pd.DataFrame, horizon: int = 14, model_path: Path = Path("models/model.joblib")) -> pd.DataFrame:
    model, metadata, features = load_model(model_path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").asfreq("D")

    last_date = df.index.max()
    preds = []
    history = df[["y"]].copy()

    for h in range(horizon):
        next_date = last_date + pd.Timedelta(days=1)
        # build a single-row feature frame for next_date using history
        row = {}
        # lag features (we detect by name)
        for feat in features:
            if feat.startswith("lag_"):
                lag = int(feat.split("_")[1])
                val = history["y"].shift(lag).iloc[-1]
                row[feat] = float(val) if pd.notna(val) else 0.0
            elif feat.startswith("rolling_mean_"):
                w = int(feat.split("_")[-1])
                val = history["y"].shift(1).rolling(w, min_periods=1).mean().iloc[-1]
                row[feat] = float(val)
            elif feat.startswith("rolling_std_"):
                w = int(feat.split("_")[-1])
                val = history["y"].shift(1).rolling(w, min_periods=1).std().fillna(0).iloc[-1]
                row[feat] = float(val)
            elif feat == "dow":
                row[feat] = int(next_date.dayofweek)
            elif feat == "month":
                row[feat] = int(next_date.month)
            elif feat == "is_weekend":
                row[feat] = int(next_date.dayofweek in (5, 6))
            else:
                # unknown feature: default zero
                row[feat] = 0.0

        X_row = pd.DataFrame([row])
        yhat = model.predict(X_row)[0]
        preds.append((next_date, float(yhat)))
        # append prediction to history for next iteration
        history.loc[next_date] = [yhat]
        last_date = next_date

    res = pd.DataFrame({"date": [d for d, _ in preds], "yhat": [v for _, v in preds]})
    return res
