"""Training script.

Loads processed data, creates features (no leakage), trains a regressor
and saves the trained artifact with metadata (feature list, last_train_date).

The trainer prefers LightGBM or sklearn; falls back to a DummyRegressor
if neither is available so the pipeline remains runnable without deps.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import joblib
import pandas as pd

from .features import make_features


def _make_model():
    try:
        import lightgbm as lgb  # type: ignore

        def create():
            return lgb.LGBMRegressor(random_state=int(os.environ.get("RANDOM_SEED", 42)))

        return create
    except Exception:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore

            def create():
                return HistGradientBoostingRegressor(random_state=int(os.environ.get("RANDOM_SEED", 42)))

            return create
        except Exception:
            from sklearn.dummy import DummyRegressor  # type: ignore

            def create():
                return DummyRegressor(strategy="mean")

            return create


def run(output_dir: Path = Path("models"), horizon: int = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/processed/clean.csv")
    feat = make_features(df)
    X = feat[[c for c in feat.columns if c not in ["date", "y"]]]
    y = feat["y"]

    ModelFactory = _make_model()
    model = ModelFactory()
    model.fit(X, y)

    metadata = {
        "features": list(X.columns),
        "last_train_date": str(pd.to_datetime(df["date"]).max().date()),
        "horizon": int(os.environ.get("HORIZON", 14)) if horizon is None else int(horizon),
    }

    pack = {"model": model, "metadata": metadata}
    joblib.dump(pack, output_dir / "model.joblib")
    # also write a small JSON manifest
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f)
    print(f"Saved model and metadata to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--horizon", type=int, default=None)
    args = parser.parse_args()
    if args.run:
        run(output_dir=Path(args.out), horizon=args.horizon)


if __name__ == "__main__":
    main()
