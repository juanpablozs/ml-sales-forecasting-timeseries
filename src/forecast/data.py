"""Data ingestion and processing utilities.

Capabilities:
- load from URL or local CSV
- fallback to bundled sample or synthetic generator
- parse dates, reindex to daily frequency and fill missing days
- optional simple outlier clipping
- save processed CSV to `data/processed/clean.csv`
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _generate_synthetic(start: Optional[pd.Timestamp] = None, periods: int = 730) -> pd.DataFrame:
    if start is None:
        end = pd.Timestamp.today().normalize()
        start = end - pd.Timedelta(days=periods - 1)
    idx = pd.date_range(start=start, periods=periods, freq="D")
    rng = np.random.default_rng( int(os.environ.get("RANDOM_SEED", 42)) )
    trend = np.linspace(50, 200, periods)
    seasonal = 20 * np.sin(2 * np.pi * idx.dayofyear / 7)  # weekly-ish
    noise = rng.normal(0, 10, periods)
    y = trend + seasonal + noise
    return pd.DataFrame({"date": idx, "y": y})


def _read_source(source_path: Optional[Path], source_url: Optional[str]) -> pd.DataFrame:
    if source_path and source_path.exists():
        df = pd.read_csv(source_path)
        print(f"Loaded data from {source_path}")
        return df
    if source_url:
        try:
            df = pd.read_csv(source_url)
            print(f"Downloaded data from {source_url}")
            return df
        except Exception as e:
            print(f"Failed to download from url: {e}")
    sample = Path("data/raw/sample_sales.csv")
    if sample.exists():
        df = pd.read_csv(sample)
        print(f"Loaded sample data from {sample}")
        return df
    print("No source found — generating synthetic dataset.")
    return _generate_synthetic()


def process(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "y",
    fill_method: str = "ffill",
    clip_outliers: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in data")
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in data")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col]].drop_duplicates(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_idx)

    if fill_method == "ffill":
        df[value_col] = df[value_col].ffill().fillna(0)
    elif fill_method == "zero":
        df[value_col] = df[value_col].fillna(0)
    elif fill_method == "interpolate":
        df[value_col] = df[value_col].interpolate().fillna(0)
    else:
        raise ValueError("fill_method must be one of ['ffill','zero','interpolate']")

    if clip_outliers:
        q1 = df[value_col].quantile(0.25)
        q3 = df[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        df[value_col] = df[value_col].clip(lower=lower, upper=upper)

    df = df.rename_axis("date").reset_index()
    df = df[["date", value_col]]
    df = df.rename(columns={value_col: "y"})
    return df


def to_csv(df: pd.DataFrame, out_path: Path = Path("data/processed/clean.csv")) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")


def to_mysql(df: pd.DataFrame, table: str = "sales") -> None:
    try:
        from sqlalchemy import create_engine

        host = os.environ.get("MYSQL_HOST", "127.0.0.1")
        port = int(os.environ.get("MYSQL_PORT", 3306))
        user = os.environ.get("MYSQL_USER", "mluser")
        password = os.environ.get("MYSQL_PASSWORD", "mlpass")
        db = os.environ.get("MYSQL_DATABASE", "ml_sales")
        uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
        engine = create_engine(uri)
        df.to_sql(table, engine, if_exists="replace", index=False)
        print(f"Wrote {len(df)} rows to {table} table in MySQL")
    except Exception as e:
        print(f"Skipping DB write: {e}")


def run(
    source_path: Optional[Path] = None,
    source_url: Optional[str] = None,
    out: Path = Path("data/processed/clean.csv"),
    date_col: str = "date",
    value_col: str = "y",
    fill_method: str = "ffill",
    clip_outliers: bool = False,
    to_db: bool = False,
):
    src = _read_source(source_path, source_url)
    # save raw original to data/raw/original.csv for traceability
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    src.to_csv(raw_dir / "original.csv", index=False)
    df = process(src, date_col=date_col, value_col=value_col, fill_method=fill_method, clip_outliers=clip_outliers)
    to_csv(df, out)
    if to_db:
        to_mysql(df)


def main():
    parser = argparse.ArgumentParser(description="Ingest and process time-series sales data")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--source-path", type=str, default=None)
    parser.add_argument("--source-url", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/processed/clean.csv")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--value-col", type=str, default="y")
    parser.add_argument("--fill", type=str, default="ffill", choices=["ffill", "zero", "interpolate"]) 
    parser.add_argument("--clip-outliers", action="store_true")
    parser.add_argument("--to-db", action="store_true")
    args = parser.parse_args()
    if args.run:
        sp = Path(args.source_path) if args.source_path else None
        run(source_path=sp, source_url=args.source_url, out=Path(args.out), date_col=args.date_col, value_col=args.value_col, fill_method=args.fill, clip_outliers=args.clip_outliers, to_db=args.to_db)


if __name__ == "__main__":
    main()
