"""Walk-forward backtesting utilities.

Performs rolling-origin evaluation with horizon H and step size H.
Compares ML model vs naive and seasonal-naive baselines, computes MAE/RMSE/sMAPE,
saves per-fold plots and a summary report under `reports/`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except Exception:
    _PLOTTING_AVAILABLE = False
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import make_features
from .baselines import naive_forecast, seasonal_naive
from .train import _make_model


def smape(a: np.ndarray, f: np.ndarray) -> float:
    return 100.0 * np.mean(2.0 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-8))


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "smape": float(smape(y_true, y_pred)),
    }


def run(horizon: int = None, step: int = None, out_dir: Path = Path("reports")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("data/processed/clean.csv")
    horizon = int(horizon or os.environ.get("HORIZON", 14))
    step = int(step or horizon)

    feat = make_features(df)
    n = len(feat)
    # start training after initial warmup (60% of data)
    initial_train = max(int(0.6 * n), 3)

    folds: List[Dict] = []
    metrics_rows = []

    # determine train_end indices for folds
    candidate_train_ends = list(range(initial_train, n - horizon + 1, step))

    # handle small datasets: create single fold if none produced above
    if len(candidate_train_ends) == 0:
        train_end = max(int(0.8 * n), 1)
        if train_end >= n:
            train_end = max(n - 1, 1)
        candidate_train_ends = [train_end]

    # Walk-forward
    for train_end in candidate_train_ends:
        train = feat.iloc[:train_end]
        # ensure we have a validation window
        val = feat.iloc[train_end : train_end + horizon]
        if len(val) == 0:
            # try to make at least one-day validation by moving train_end earlier
            train_end = max(train_end - 1, 1)
            val = feat.iloc[train_end : train_end + horizon]

        X_train = train[[c for c in train.columns if c not in ["date", "y"]]]
        y_train = train["y"].values
        X_val = val[[c for c in val.columns if c not in ["date", "y"]]]
        y_val = val["y"].values

        # train model for this fold
        ModelFactory = _make_model()
        model = ModelFactory()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # baselines using raw series up to cutoff
        cutoff_idx = train_end - 1
        series_up_to_cutoff = feat.iloc[: train_end]["y"].reset_index(drop=True)
        h = len(val)
        if h == 0:
            continue
        naive_preds = naive_forecast(series_up_to_cutoff, h)
        seasonal_preds = seasonal_naive(series_up_to_cutoff, h)

        metrics_ml = evaluate_fold(y_val, y_pred)
        metrics_naive = evaluate_fold(y_val, np.array(naive_preds))
        metrics_seasonal = evaluate_fold(y_val, np.array(seasonal_preds))

        metrics_rows.append(
            {
                "train_end_idx": int(train_end),
                "train_end_date": str(feat.iloc[train_end - 1]["date"]),
                "mae_ml": metrics_ml["mae"],
                "rmse_ml": metrics_ml["rmse"],
                "smape_ml": metrics_ml["smape"],
                "mae_naive": metrics_naive["mae"],
                "mae_seasonal": metrics_seasonal["mae"],
            }
        )

        # save fold predictions for plotting
        fold_df = pd.DataFrame(
            {
                "date": val["date"].values,
                "y_true": y_val,
                "y_pred": y_pred,
                "naive": naive_preds,
                "seasonal": seasonal_preds,
            }
        )
        fold_name = f"fold_{train_end}"
        fold_df.to_csv(out_dir / f"{fold_name}_preds.csv", index=False)

        # plot (if possible)
        if _PLOTTING_AVAILABLE:
            plt.figure(figsize=(8, 4))
            sns.lineplot(x="date", y="y_true", data=fold_df, label="actual")
            sns.lineplot(x="date", y="y_pred", data=fold_df, label="model")
            sns.lineplot(x="date", y="naive", data=fold_df, label="naive", linestyle="--")
            plt.xticks(rotation=30)
            plt.title(f"Backtest {fold_name} (train_end={feat.iloc[train_end-1]['date']})")
            plt.tight_layout()
            plt.savefig(out_dir / f"{fold_name}.png")
            plt.close()
            plot_path = str(out_dir / f"{fold_name}.png")
        else:
            plot_path = ""

        folds.append({"fold": fold_name, "preds_csv": str(out_dir / f"{fold_name}_preds.csv"), "plot": plot_path})

    # summary metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "backtest_summary.csv", index=False)

    # human-friendly markdown summary
    md = []
    md.append("# Backtest Summary")
    md.append("")
    md.append("## Aggregate metrics")
    md.append("")
    md.append(metrics_df.describe().to_string())
    md.append("")
    md.append("## Per-fold metrics")
    md.append("")
    md.append(metrics_df.to_string(index=False))
    md.append("")
    md.append("## Fold plots")
    md.append("")
    for f in folds:
        md.append(f"![{f['fold']}]({f['plot']})")

    with open(out_dir / "backtest_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print(f"Backtest complete. Reports saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", type=str, default="reports")
    args = parser.parse_args()
    if args.run:
        run(horizon=args.horizon, step=args.step, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
