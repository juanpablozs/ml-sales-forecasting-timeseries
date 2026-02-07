"""Training script (placeholder).

Will load processed data, create features, train a model and save artifact.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import make_features


def run(output_dir: Path = Path("models")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/processed/clean.csv")
    feat = make_features(df)
    X = feat[[c for c in feat.columns if c not in ["date", "y"]]]
    y = feat["y"]
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    joblib.dump({"model": model, "features": list(X.columns)}, output_dir / "model.joblib")
    print(f"Saved model to {output_dir / 'model.joblib'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()


if __name__ == "__main__":
    main()
