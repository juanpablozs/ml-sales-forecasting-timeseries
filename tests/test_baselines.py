import pandas as pd
from src.forecast.baselines import naive_forecast, seasonal_naive


def test_naive_forecast_length_and_value():
    s = pd.Series([10, 12, 15])
    preds = naive_forecast(s, horizon=4)
    assert len(preds) == 4
    assert all(p == 15 for p in preds)


def test_seasonal_naive_weekly():
    # create 14 days of data with pattern 1..7 repeated twice
    vals = list(range(1, 8)) + list(range(1, 8))
    s = pd.Series(vals)
    preds = seasonal_naive(s, horizon=5, season=7)
    # expected to use last 7 values (1..7) and repeat
    expected = [1, 2, 3, 4, 5]
    assert preds == expected
