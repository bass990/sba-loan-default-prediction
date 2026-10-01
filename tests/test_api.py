"""End-to-end tests for the SBA FastAPI service.

These hit the live scoring pipeline (load real artifacts → run inference)
because the scoring function is tiny and fast enough that mocking it would
hide more than it would speed up. The training data SHA + artifact files are
the contract under test; if either drifts, these tests should catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    """One TestClient per module — the lifespan startup loads artifacts once."""
    with TestClient(app) as c:
        yield c


def _valid_loan(index: int = 1) -> dict:
    """A real holdout row (index 0 from the holdout CSV, re-indexed)."""
    return {
        "index": index,
        "City": "GARDENA",
        "State": "CA",
        "Zip": 90249,
        "Bank": "ONEUNITED BANK",
        "BankState": "MA",
        "NAICS": 0,
        "NoEmp": 6,
        "NewExist": 2.0,
        "CreateJob": 0,
        "RetainedJob": 0,
        "FranchiseCode": 1,
        "UrbanRural": 0,
        "RevLineCr": "0",
        "LowDoc": "Y",
        "DisbursementGross": 50000.0,
        "BalanceGross": 0.0,
        "GrAppv": 50000.0,
        "SBA_Appv": 40000.0,
    }


def test_health_after_startup(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info_reports_honest_ensemble_weights(client: TestClient) -> None:
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["n_features"] == len(body["feature_names"])
    assert body["n_features"] >= 30  # 16 raw + 15 engineered = 31
    # The honest-disclosure caveat — weight search selected model_1 only.
    assert body["ensemble_weights"]["model_1"] == pytest.approx(1.0)
    assert body["ensemble_weights"]["model_2"] == pytest.approx(0.0)
    assert "model_1 alone" in body["honest_disclosure"]
    # Headline number must still match the trained model card.
    assert 0.55 < body["aucpr_test"] < 0.58


def test_predict_single_loan_returns_calibrated_probs(client: TestClient) -> None:
    r = client.post("/predict", json=_valid_loan())
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    pred = body["predictions"][0]
    assert pred["index"] == 1
    assert pred["label"] in (0, 1)
    assert 0.0 <= pred["probability_0"] <= 1.0
    assert 0.0 <= pred["probability_1"] <= 1.0
    assert pred["probability_0"] + pred["probability_1"] == pytest.approx(1.0, abs=1e-6)


def test_predict_batch_preserves_input_order(client: TestClient) -> None:
    loans = [_valid_loan(index=i) for i in range(5)]
    r = client.post("/predict-batch", json={"loans": loans})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 5
    assert [p["index"] for p in body["predictions"]] == [0, 1, 2, 3, 4]


def test_predict_rejects_negative_loan_amount(client: TestClient) -> None:
    bad = _valid_loan()
    bad["GrAppv"] = -1.0
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    bad = _valid_loan()
    del bad["State"]
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_rejects_extra_field(client: TestClient) -> None:
    """extra='forbid' on the schema — unknown fields are an error, not silently dropped."""
    bad = _valid_loan()
    bad["sneaky_unknown_column"] = "x"
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_batch_rejects_empty_list(client: TestClient) -> None:
    r = client.post("/predict-batch", json={"loans": []})
    assert r.status_code == 422
