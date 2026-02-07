"""Walk-forward backtesting utilities (placeholder)."""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from .features import make_features


def smape(a, f):
    return 100.0 * np.mean(2.0 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-8))


def run(horizon: int = 14):
    df = pd.read_csv("data/processed/clean.csv")
    feat = make_features(df)
    # Very simple single-fold: train on first 70%, test last horizon
    n = len(feat)
    train = feat.iloc[: int(n * 0.8)]
    test = feat.iloc[int(n * 0.8) :]
    X_train = train[[c for c in train.columns if c not in ["date", "y"]]]
    y_train = train["y"]
    X_test = test[[c for c in test.columns if c not in ["date", "y"]]]
    y_test = test["y"]
    model_pack = joblib.load("models/model.joblib")
    model = model_pack["model"]
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    s = smape(y_test.values, preds)
    print(f"MAE={mae:.4f} RMSE={rmse:.4f} sMAPE={s:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()


if __name__ == "__main__":
    main()
