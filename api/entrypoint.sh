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
  # If model exists, compare modification times to processed data; skip training
  if [ -f "${DATA}" ]; then
    model_mtime=$(stat -c %Y "${MODEL}" 2>/dev/null || stat -f %m "${MODEL}" 2>/dev/null || echo 0)
    data_mtime=$(stat -c %Y "${DATA}" 2>/dev/null || stat -f %m "${DATA}" 2>/dev/null || echo 0)
    if [ "${model_mtime}" -ge "${data_mtime}" ]; then
      echo "Model is newer than processed data; skipping training."
    else
      echo "Model older than processed data; retraining model..."
      python -m src.forecast.train --run
    fi
  else
    echo "Processed data not present to compare timestamps; skipping retrain."
  fi
fi

echo "Starting uvicorn..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
