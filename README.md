# ML Sales Forecasting (e2e)

This repository implements an end-to-end, reproducible pipeline for daily sales forecasting. It includes ingestion, feature engineering, baseline models, ML training, walk-forward backtesting, reporting, and a FastAPI inference service.

Next steps:

- Run `make setup` to install dependencies.
- Run `make ingest` to download/process data.
- Run `make train` to train the model and save artifacts.
- Run `make backtest` to run the rolling evaluation and produce reports.
- Run `make serve` to start the FastAPI service.
# ml-sales-forecasting-timeseries