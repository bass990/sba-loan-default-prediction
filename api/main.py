"""FastAPI service wrapping the SBA loan-default XGBoost ensemble.

Design notes
------------
* The scoring logic lives in `score_model.score()` at the project root and is
  reused as-is. The API is a thin validation + transport + monitoring layer;
  no modeling decisions are duplicated here.
* Model artifacts are loaded once per process (`score_model.load_artifacts`
  is cached) and touched at startup via the lifespan context, so a missing
  file fails the container, not the first request.
* /model-info reports the actual ensemble weights (1.0 / 0.0: model_1 alone
  won the weight search) and, when artifacts/eval_report.json exists, the
  held-out calibration and worst-slice numbers. Surface what is true.
* /drift and the batch drift summary compare incoming feature distributions
  against artifacts/reference_stats.json with PSI (api/monitoring.py).
* Every response carries X-Request-ID and X-Process-Time-Ms; one structured
  log line per scoring request.
* CORS is open to all origins because this is a demo API; lock it down with
  SBA_API_CORS_ORIGINS in a real deployment.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Make the project root importable so we can reuse score_model.score().
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.monitoring import MIN_ROWS_FOR_DRIFT, drift_report, load_reference  # noqa: E402
from api.schemas import (  # noqa: E402
    BatchRequest,
    DriftResponse,
    HealthResponse,
    LoanRequest,
    ModelInfo,
    Prediction,
    PredictResponse,
)
from score_model import _clean_df, _engineer_features, load_artifacts  # noqa: E402
from score_model import score as score_fn  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REFERENCE_PATH = ARTIFACTS_DIR / "reference_stats.json"
EVAL_REPORT_PATH = ARTIFACTS_DIR / "eval_report.json"
API_VERSION = "1.1.0"
MAX_BATCH = 10_000

log = logging.getLogger("sba_api")
logging.basicConfig(level=os.getenv("SBA_API_LOG_LEVEL", "INFO"), format="%(message)s")

# Warmed state populated by the lifespan handler.
_state: dict = {"model_loaded": False, "metadata": None, "reference": None, "eval_report": None, "started_at": None}


def _make_warmup_frame() -> pd.DataFrame:
    """A single synthetic row matching the holdout schema, used only at startup."""
    return pd.DataFrame([{
        "index": -1, "City": "WARMUP", "State": "CA", "Zip": 90001, "Bank": "WARMUP_BANK", "BankState": "CA",
        "NAICS": 0, "NoEmp": 1, "NewExist": 1.0, "CreateJob": 0, "RetainedJob": 0, "FranchiseCode": 1, "UrbanRural": 0,
        "RevLineCr": "N", "LowDoc": "N", "DisbursementGross": 10000.0, "BalanceGross": 0.0, "GrAppv": 10000.0, "SBA_Appv": 5000.0,
    }])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Touch artifacts at boot so a missing file fails the container, not the first request."""
    art = load_artifacts()
    score_fn(_make_warmup_frame())  # encoders + boosters warm, JIT path exercised
    _state["metadata"] = art["metadata"]
    _state["reference"] = load_reference(REFERENCE_PATH)
    _state["eval_report"] = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8")) if EVAL_REPORT_PATH.exists() else None
    _state["model_loaded"] = True
    _state["started_at"] = time.time()
    log.info(json.dumps({"event": "startup", "features": len(art["metadata"]["feature_names"]),
                         "drift_reference": _state["reference"] is not None, "eval_report": _state["eval_report"] is not None}))
    yield


app = FastAPI(
    title="SBA Loan Default Prediction API",
    version=API_VERSION,
    description=(
        "XGBoost classifier predicting default risk on SBA-guaranteed loans. "
        "Trained on the SBA Project 2 dataset; held-out AUCPR 0.567 (baseline 0.18). "
        "See /model-info for the honest model card and /drift for input monitoring."
    ),
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("SBA_API_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    if request.url.path.startswith(("/predict", "/drift")):
        log.info(json.dumps({"event": "request", "id": rid, "path": request.url.path, "status": response.status_code, "ms": round(ms, 1)}))
    return response


def _require_loaded() -> None:
    if not _state["model_loaded"]:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Model not loaded yet")


def _to_dataframe(loans: list[LoanRequest]) -> pd.DataFrame:
    return pd.DataFrame([loan.model_dump() for loan in loans])


def _to_predictions(df: pd.DataFrame) -> list[Prediction]:
    return [Prediction(index=int(r["index"]), label=int(r["label"]), probability_0=float(r["probability_0"]),
                       probability_1=float(r["probability_1"])) for _, r in df.iterrows()]


def _score(df_in: pd.DataFrame) -> pd.DataFrame:
    try:
        return score_fn(df_in)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _drift(df_in: pd.DataFrame, probabilities) -> dict | None:
    ref = _state["reference"]
    if ref is None or len(df_in) < MIN_ROWS_FOR_DRIFT:
        return None
    feats = _engineer_features(_clean_df(df_in))
    return drift_report(feats, ref, probabilities)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """A person opening the API in a browser lands on the scoring UI when it is bundled, else the docs."""
    return RedirectResponse(url="/ui" if _UI_MOUNTED else "/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok" if _state["model_loaded"] else "loading", model_loaded=_state["model_loaded"],
                          version=API_VERSION, drift_reference_loaded=_state["reference"] is not None,
                          uptime_s=round(time.time() - _state["started_at"], 1) if _state["started_at"] else 0.0)


@app.get("/model-info", response_model=ModelInfo, tags=["meta"])
def model_info() -> ModelInfo:
    _require_loaded()
    meta = _state["metadata"]
    rep = _state["eval_report"] or {}
    cal = rep.get("calibration") or {}
    return ModelInfo(
        model_name="SBA Loan Default: XGBoost ensemble",
        model_version=API_VERSION,
        framework="xgboost",
        n_features=len(meta["feature_names"]),
        feature_names=meta["feature_names"],
        decision_threshold=meta["threshold_ensemble"],
        aucpr_test=meta["aucpr_test_ens"],
        class_imbalance_scale_pos_weight=meta["spw"],
        ensemble_weights={"model_1": meta["weight_model_1"], "model_2": meta["weight_model_2"]},
        honest_disclosure=(
            "The two-model weight search selected model_1 alone (weights 1.0 / 0.0). Reported metrics reflect model_1 on the "
            "held-out test set; the second booster did not contribute and is retained only for reproducibility of the search. "
            "Probabilities are ranked well but over-state default risk because of scale_pos_weight=4.71; see calibration."
        ),
        held_out_evaluation=rep.get("headline"),
        calibration={"brier": cal.get("brier"), "ece": cal.get("ece"), "note": cal.get("note")} if cal else None,
        worst_slice_by_false_negative_rate=rep.get("worst_slice_by_fnr"),
        cost_weighted_threshold=(rep.get("cost_weighted_threshold") or {}).get("best_threshold"),
        training_data_vintage="SBA 7(a) public loan-level file; loans through 2014",
        drift_reference={"source": _state["reference"].get("source"), "n_rows": _state["reference"].get("n_rows")} if _state["reference"] else None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(loan: LoanRequest) -> PredictResponse:
    """Score a single loan record. Returns label + class probabilities."""
    _require_loaded()
    return PredictResponse(predictions=_to_predictions(_score(_to_dataframe([loan]))))


@app.post("/predict-batch", response_model=PredictResponse, tags=["predict"])
def predict_batch(req: BatchRequest) -> PredictResponse:
    """Score 1..10,000 loans in one round-trip; predictions return in input order.
    Batches of 200+ rows also get a PSI drift summary against the training reference."""
    _require_loaded()
    df_in = _to_dataframe(req.loans)
    df_out = _score(df_in)
    return PredictResponse(predictions=_to_predictions(df_out), drift=_drift(df_in, df_out["probability_1"].to_numpy()))


@app.post("/drift", response_model=DriftResponse, tags=["monitoring"])
def drift(req: BatchRequest) -> DriftResponse:
    """PSI of a batch's engineered features (and score distribution) vs the training reference."""
    _require_loaded()
    if _state["reference"] is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "No drift reference (run scripts/build_reference.py)")
    if len(req.loans) < MIN_ROWS_FOR_DRIFT:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Drift needs at least {MIN_ROWS_FOR_DRIFT} rows; got {len(req.loans)}")
    df_in = _to_dataframe(req.loans)
    df_out = _score(df_in)
    return DriftResponse(**_drift(df_in, df_out["probability_1"].to_numpy()))


# The Gradio scoring UI (huggingface-spaces/app.py, the same file the Hugging Face Space runs) is mounted at /ui
# when the file and gradio are present. Registered last so every API route above keeps precedence.
_UI_MOUNTED = False
_UI_APP = next((p for p in (Path(__file__).resolve().parent.parent / "huggingface-spaces" / "app.py",) if p.exists()), None)
if _UI_APP is not None:
    try:
        import gradio as gr  # noqa: PLC0415

        _spec = importlib.util.spec_from_file_location("sba_gradio_app", _UI_APP)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        app = gr.mount_gradio_app(app, _mod.demo, path="/ui")
        _UI_MOUNTED = True
    except Exception as exc:  # noqa: BLE001 - the API must come up even if the UI cannot
        logging.getLogger("sba_api").warning("scoring UI not mounted: %s", exc)
