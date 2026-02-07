from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
from pathlib import Path

from src.forecast.predict import forecast as _forecast, load_model

app = FastAPI(title="ml-sales-forecasting-e2e")


class ForecastRequest(BaseModel):
    horizon: Optional[int] = 14
    start_date: Optional[str] = None


class ForecastPoint(BaseModel):
    date: str
    yhat: float


class ForecastResponse(BaseModel):
    predictions: List[ForecastPoint]


@app.on_event("startup")
def load_artifacts():
    model_file = Path("models/model.joblib")
    if model_file.exists():
        try:
            app.state.model_pack = load_model(model_file)
        except Exception:
            app.state.model_pack = None
    else:
        app.state.model_pack = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    pack = getattr(app.state, "model_pack", None)
    if pack is None:
        raise HTTPException(status_code=404, detail="Model not found")
    meta = pack.get("metadata", {})
    return {"model": True, "metadata": meta}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    df = pd.read_csv("data/processed/clean.csv")
    if req.start_date:
        try:
            cutoff = pd.to_datetime(req.start_date)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
        df = df[pd.to_datetime(df["date"]) <= cutoff]
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available up to start_date")

    model_pack = getattr(app.state, "model_pack", None)
    if model_pack is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded")

    res = _forecast(df, horizon=req.horizon, model_pack=model_pack)
    return {"predictions": [{"date": str(d.date()), "yhat": float(y)} for d, y in zip(pd.to_datetime(res["date"]), res["yhat"]) ]}
