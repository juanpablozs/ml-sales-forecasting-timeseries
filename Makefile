PYTHON=python
REQ=requirements.txt

.PHONY: setup ingest train backtest serve lint

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQ)

ingest:
	$(PYTHON) -m src.forecast.data --run

train:
	$(PYTHON) -m src.forecast.train --run

backtest:
	$(PYTHON) -m src.forecast.backtest --run

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff check .
