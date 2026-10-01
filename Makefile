.PHONY: install test lint api reference evaluate golden docker-build docker-run docker-smoke clean help

PYTHON ?= python

help:
	@echo "SBA loan default: make targets"
	@echo "  install        pip install -r requirements-api.txt + dev tools (pytest, httpx, ruff)"
	@echo "  test           pytest tests/ (API contract, golden predictions, adversarial inputs, drift)"
	@echo "  lint           ruff check api scripts tests score_model.py"
	@echo "  api            uvicorn api.main:app --reload --port 8000"
	@echo "  reference      freeze the training-data feature distribution for drift (artifacts/reference_stats.json)"
	@echo "  evaluate       held-out calibration / slices / cost-weighted threshold (artifacts/eval_report.json)"
	@echo "  golden         re-freeze tests/golden_predictions.json after a retrain"
	@echo "  docker-build   build the lean inference image"
	@echo "  docker-run     run it on :8000"
	@echo "  docker-smoke   build, run, curl /health /model-info /predict, stop"

install:
	$(PYTHON) -m pip install -r requirements-api.txt pytest httpx ruff

test:
	$(PYTHON) -m pytest tests/ -q

lint:
	$(PYTHON) -m ruff check api scripts tests score_model.py

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

reference:
	$(PYTHON) scripts/build_reference.py

evaluate:
	$(PYTHON) scripts/evaluate.py

golden:
	$(PYTHON) scripts/freeze_golden.py

docker-build:
	docker build -t sba-api .

docker-run: docker-build
	docker run --rm -p 8000:8000 sba-api

docker-smoke: docker-build
	docker run -d --name sba-smoke -p 8000:8000 sba-api
	sleep 8
	curl -sf http://localhost:8000/health
	curl -sf http://localhost:8000/model-info | head -c 300
	curl -sf -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @tests/example_request.json
	docker rm -f sba-smoke

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
