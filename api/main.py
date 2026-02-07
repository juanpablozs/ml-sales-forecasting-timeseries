from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd

app = FastAPI(title="ml-sales-forecasting-e2e")


class ForecastRequest(BaseModel):
    horizon: Optional[int] = 14


class ForecastPoint(BaseModel):
    date: str
    yhat: float


class ForecastResponse(BaseModel):
    predictions: List[ForecastPoint]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    # Simple implementation: load processed data and use predict.forecast
    df = pd.read_csv("data/processed/clean.csv")
    from src.forecast.predict import forecast as _forecast

    res = _forecast(df, horizon=req.horizon)
    return {"predictions": [ {"date": str(d.date()), "yhat": float(y)} for d,y in zip(pd.to_datetime(res['date']), res['yhat']) ]}
