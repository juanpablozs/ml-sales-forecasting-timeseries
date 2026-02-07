"""Run quick checks and API tests without requiring pytest installation.

This script executes small test functions to validate core functionality.
"""
import sys
from pathlib import Path

proj_root = str(Path(__file__).resolve().parents[1])
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from tests.run_quick_checks import check_features, check_baselines
from fastapi.testclient import TestClient
from api.main import app


def run_api_smoke():
    client = TestClient(app)
    r = client.get('/health')
    assert r.status_code == 200
    r2 = client.post('/forecast', json={'horizon': 3})
    assert r2.status_code == 200


if __name__ == '__main__':
    check_features()
    check_baselines()
    print('Quick checks passed')
    run_api_smoke()
    print('API smoke tests passed')
