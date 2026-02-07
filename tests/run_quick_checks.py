import sys
from pathlib import Path
proj_root = str(Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.forecast.features import make_features
from src.forecast.baselines import naive_forecast, seasonal_naive
import pandas as pd

def check_features():
    idx = pd.date_range(end=pd.Timestamp('2020-01-10'), periods=10, freq='D')
    df = pd.DataFrame({'date': idx, 'y': range(10)})
    feat = make_features(df)
    assert (feat['lag_1'] <= feat['y']).all()

def check_baselines():
    s = pd.Series([10, 12, 15])
    preds = naive_forecast(s, horizon=4)
    assert len(preds) == 4 and all(p == 15 for p in preds)
    vals = list(range(1, 8)) + list(range(1, 8))
    s2 = pd.Series(vals)
    preds2 = seasonal_naive(s2, horizon=5, season=7)
    assert preds2 == [1,2,3,4,5]

if __name__ == '__main__':
    check_features()
    check_baselines()
    print('Quick checks passed')
