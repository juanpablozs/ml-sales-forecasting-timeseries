# ML Sales Forecasting (e2e)

This repository implements an end-to-end, reproducible pipeline for daily sales forecasting. It includes ingestion, feature engineering, baseline models, ML training, walk-forward backtesting, reporting, and a FastAPI inference service.

Quick start (local):

- Install dependencies:

```bash
make setup
```

- Ingest data (loads sample or downloads if URL provided):

```bash
make ingest
```

- Train model:

```bash
make train
```

- Run backtest and generate reports:

```bash
make backtest
```

- Generate a final forecast CSV:

```bash
python -m src.forecast.predict --run
```

- Serve API locally:

```bash
make serve
```

Docker (quick):

```bash
docker-compose build --pull --no-cache
docker-compose up --force-recreate
```

API endpoints:

- `GET /health` — health check
- `GET /model-info` — model metadata (features, last_train_date)
- `POST /forecast` — body: `{ "horizon": 14, "start_date": "YYYY-MM-DD" }`

Example curl:

```bash
curl -X POST http://localhost:8000/forecast -H "Content-Type: application/json" -d '{"horizon":14}'
```

Design notes:

- Business goal: forecast daily sales to support inventory and staffing.
- Evaluation: walk-forward (rolling-origin) backtest to avoid leakage.
- Baselines: naive (y_t+1 = y_t) and seasonal-naive (y_t+7 = y_t).
- Metrics: MAE, RMSE, sMAPE (symmetric and stable for values near zero).

For more details see the `src/` modules and `reports/` produced by `make backtest`.
# ml-sales-forecasting-timeseries