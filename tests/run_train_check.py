import sys
from pathlib import Path
proj_root = str(Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.forecast import train, predict
import pandas as pd
from pathlib import Path

def main():
    # ensure processed data exists
    p = Path('data/processed/clean.csv')
    if not p.exists():
        print('Processed data not found. Run ingestion first.')
        return
    train.run(output_dir=Path('models'))
    assert (Path('models') / 'model.joblib').exists()
    df = pd.read_csv(p)
    res = predict.forecast(df, horizon=5)
    assert len(res) == 5
    print('Train and predict checks passed')

if __name__ == '__main__':
    main()
