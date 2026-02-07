#!/usr/bin/env bash
set -euo pipefail

# On container start: ensure data and model exist. If model missing, run ingest+train.
echo "Entry point: check for processed data and model artifacts"

APP_DIR="/app"
DATA="${APP_DIR}/data/processed/clean.csv"
MODEL="${APP_DIR}/models/model.joblib"

if [ ! -f "${DATA}" ]; then
  echo "Processed data not found. Running ingest..."
  python -m src.forecast.data --run
else
  echo "Processed data found: ${DATA}"
fi

if [ ! -f "${MODEL}" ]; then
  echo "Model artifact not found. Training model..."
  python -m src.forecast.train --run
else
  echo "Model artifact found: ${MODEL}"
fi

echo "Starting uvicorn..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
