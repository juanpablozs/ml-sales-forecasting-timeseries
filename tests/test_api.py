from fastapi.testclient import TestClient
from api.main import app
import pandas as pd


client = TestClient(app)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200


def test_forecast_length():
    r = client.post('/forecast', json={'horizon': 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data['predictions']) == 5
