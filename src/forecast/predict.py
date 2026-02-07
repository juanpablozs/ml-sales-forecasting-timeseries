"""Prediction utilities for producing H-day forecasts.

This implements iterative forecasting when features use lags/rolling stats:
- load trained model and feature list
- iteratively predict day t+1, append predicted y, recompute features and continue
"""
from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
import os


def load_model(model_path: Path = Path("models/model.joblib")):
    pack = joblib.load(model_path)
    return pack


def forecast(
    df: pd.DataFrame,
    horizon: int = 14,
    model_path: Path = Path("models/model.joblib"),
    model_pack: dict | None = None,
) -> pd.DataFrame:
    if model_pack is None:
        model_pack = load_model(model_path)
    model = model_pack["model"]
    metadata = model_pack.get("metadata", {})
    features = metadata.get("features", [])
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


def run(
    source_path: Path = Path("data/processed/clean.csv"),
    horizon: int = None,
    out_dir: Path = Path("reports/forecasts"),
    model_path: Path = Path("models/model.joblib"),
    to_db: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source_path)
    horizon = int(horizon or int(os.environ.get("HORIZON", 14)))
    model_pack = load_model(model_path)
    preds = forecast(df, horizon=horizon, model_pack=model_pack)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"forecast_{ts}.csv"
    preds.to_csv(out_path, index=False)
    print(f"Saved forecast to {out_path}")
    if to_db:
        try:
            from .data import to_mysql

            to_mysql(preds, table="forecasts")
        except Exception as e:
            print(f"Failed to write forecasts to DB: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--out", type=str, default="reports/forecasts")
    parser.add_argument("--to-db", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(horizon=args.horizon, out_dir=Path(args.out), to_db=args.to_db)


if __name__ == "__main__":
    main()
