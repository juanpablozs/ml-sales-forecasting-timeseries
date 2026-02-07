"""
Data ingestion and processing utilities.

The real ingestion script will download or load a dataset, parse dates,
reindex to daily frequency and save processed CSV to data/processed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def run(output_path: Path = Path("data/processed/clean.csv")) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder: generate synthetic series if no dataset available
    idx = pd.date_range(end=pd.Timestamp.today(), periods=365, freq="D")
    df = pd.DataFrame({"date": idx, "y": (100 + (idx.dayofyear * 0.1)).astype(float)})
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()


if __name__ == "__main__":
    main()
