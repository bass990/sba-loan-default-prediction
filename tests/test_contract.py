"""Scoring-contract, regression and monitoring tests (no network, real artifacts).

* golden predictions: eight holdout rows must score to the same probabilities
  the shipped artifacts produced when this file was frozen (tolerance 1e-4).
  Any silent change to feature engineering, encoders or the booster fails here.
* model-card pins: the numbers the README quotes are read from the artifacts,
  so a retrain that moves them fails the build until the README is updated.
* adversarial inputs: unseen categories, dirty flags, extreme values, string
  numbers, oversized batches, NaN-free contract.
* drift: PSI is ~0 for a sample drawn from the reference and large for a
  shifted sample; the endpoint refuses batches too small to bin.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import score_model as sm  # noqa: E402
from api import monitoring  # noqa: E402
from api.main import app  # noqa: E402

GOLDEN = json.loads((PROJECT_ROOT / "tests" / "golden_predictions.json").read_text(encoding="utf-8"))["rows"]
HOLDOUT = PROJECT_ROOT / "data" / "SBA_loans_project_2_holdout_students_valid.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def _loan(**over) -> dict:
    base = dict(GOLDEN[0]["input"])
    base.update(over)
    return base


# ── regression: golden predictions ─────────────────────────────────────────

def test_golden_predictions_reproduce():
    df = pd.DataFrame([r["input"] for r in GOLDEN])
    out = sm.score(df)
    for r, p in zip(GOLDEN, out["probability_1"], strict=True):
        assert abs(float(p) - r["probability_1"]) < 1e-4, f"row {r['input']['index']} drifted: {p} vs {r['probability_1']}"
    assert list(out["label"]) == [r["label"] for r in GOLDEN]


def test_model_card_numbers_match_artifacts():
    meta = sm.load_artifacts()["metadata"]
    assert round(meta["aucpr_test_ens"], 4) == 0.5665
    assert round(meta["threshold_ensemble"], 2) == 0.66
    assert round(meta["spw"], 2) == 4.71
    assert len(meta["feature_names"]) == 31
    assert (meta["weight_model_1"], meta["weight_model_2"]) == (1.0, 0.0)


def test_artifacts_load_once():
    sm.load_artifacts.cache_clear()
    a = sm.load_artifacts()
    b = sm.load_artifacts()
    assert a is b and sm.load_artifacts.cache_info().hits >= 1


# ── adversarial inputs ──────────────────────────────────────────────────────

@pytest.mark.parametrize("over", [
    {"State": "ZZ", "BankState": "QQ"},                 # unseen categories -> UNSEEN bucket
    {"RevLineCr": "maybe", "LowDoc": "T"},              # dirty flags -> U
    {"NAICS": 0, "NewExist": 0.0},                      # coded-missing values
    {"Zip": 0, "NoEmp": 0, "CreateJob": 0, "RetainedJob": 0},
    {"DisbursementGross": 5e9, "GrAppv": 5e9, "SBA_Appv": 5e9, "NoEmp": 100000},  # extreme magnitudes
    {"City": "x" * 64, "Bank": "y" * 128},
])
def test_predict_survives_edge_inputs(client, over):
    r = client.post("/predict", json=_loan(**over))
    assert r.status_code == 200, r.text
    p = r.json()["predictions"][0]
    assert 0.0 <= p["probability_1"] <= 1.0 and p["label"] in (0, 1)


def test_predict_coerces_numeric_strings_but_rejects_garbage(client):
    r = client.post("/predict", json=_loan(NoEmp="6", GrAppv="50000.0"))
    assert r.status_code == 200
    r = client.post("/predict", json=_loan(NoEmp="six"))
    assert r.status_code == 422


def test_predict_rejects_infinite_and_null(client):
    assert client.post("/predict", json=_loan(GrAppv=None)).status_code == 422
    assert client.post("/predict", json=_loan(Zip=1_000_000)).status_code == 422
    assert client.post("/predict", json=_loan(UrbanRural=7)).status_code == 422


def test_batch_limit_enforced(client):
    loans = [_loan(index=i) for i in range(10_001)]
    r = client.post("/predict-batch", json={"loans": loans})
    assert r.status_code == 422


def test_response_headers_and_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and "X-Request-ID" in r.headers and "X-Process-Time-Ms" in r.headers
    assert r.json()["version"] and r.json()["model_loaded"]
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_model_info_carries_evaluation_when_present(client):
    body = client.get("/model-info").json()
    assert "over-state" in body["honest_disclosure"]
    if (PROJECT_ROOT / "artifacts" / "eval_report.json").exists():
        assert body["held_out_evaluation"]["aucpr"] == pytest.approx(body["aucpr_test"], abs=0.002)
        assert body["calibration"]["ece"] is not None
        assert body["worst_slice_by_false_negative_rate"]["false_negative_rate"] >= 0


# ── drift ───────────────────────────────────────────────────────────────────

def test_psi_zero_for_identical_and_large_for_shift():
    ref = np.array([0.1] * 10)
    assert monitoring.psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    shifted = np.array([0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.003, 0.002])
    assert monitoring.psi(ref, shifted) > 0.25


@pytest.mark.skipif(not (PROJECT_ROOT / "artifacts" / "reference_stats.json").exists(), reason="no drift reference built")
def test_drift_endpoint_on_holdout_sample_and_shifted_sample(client):
    if not HOLDOUT.exists():
        pytest.skip("holdout file not present")
    df = pd.read_csv(HOLDOUT).dropna().head(400)
    loans = df.to_dict(orient="records")
    r = client.post("/drift", json={"loans": loans})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 400 and len(body["features"]) >= 25
    # in-distribution sample: the large majority of features are stable
    assert body["n_shifted"] <= 3
    # now shift every loan to a 50x larger size: log_disbursement / GrAppv must move
    for loan in loans:
        for k in ("DisbursementGross", "GrAppv", "SBA_Appv"):
            loan[k] = float(loan[k]) * 50 + 1_000_000
    r2 = client.post("/drift", json={"loans": loans})
    assert r2.status_code == 200
    f2 = r2.json()["features"]
    assert f2["log_disbursement"]["status"] == "shifted" and f2["GrAppv"]["status"] == "shifted"
    assert r2.json()["score_psi"]["psi"] >= 0


def test_drift_refuses_small_batches(client):
    r = client.post("/drift", json={"loans": [_loan(index=i) for i in range(5)]})
    assert r.status_code in (422, 503)


@pytest.mark.skipif(not (PROJECT_ROOT / "artifacts" / "reference_stats.json").exists(), reason="no drift reference built")
def test_batch_predict_attaches_drift_only_for_large_batches(client):
    small = client.post("/predict-batch", json={"loans": [_loan(index=i) for i in range(3)]}).json()
    assert small["drift"] is None
    if HOLDOUT.exists():
        big = pd.read_csv(HOLDOUT).dropna().head(250).to_dict(orient="records")
        body = client.post("/predict-batch", json={"loans": big}).json()
        assert body["drift"] is not None and body["drift"]["n_rows"] == 250
