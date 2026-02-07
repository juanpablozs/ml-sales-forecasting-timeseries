import pandas as pd
from src.forecast.features import make_features


def test_make_features_no_leakage():
    idx = pd.date_range(end=pd.Timestamp('2020-01-10'), periods=10, freq='D')
    df = pd.DataFrame({'date': idx, 'y': range(10)})
    feat = make_features(df)
    # Ensure lags are shifted (no future values present)
    assert (feat['lag_1'] <= feat['y']).all()
