"""Pydantic v2 request/response schemas for the SBA loan default API.

The input fields mirror the raw Project 2 holdout format exactly. Feature
engineering (ratios, log transforms, categorical encoding) happens inside
score_model.score() — the API layer's job is to validate raw input and
hand a DataFrame to the existing scoring function.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LoanRequest(BaseModel):
    """One SBA loan record in the same shape as the training data.

    `index` is required by the scoring contract — it round-trips into the
    response so a caller batching N loans can re-join predictions to inputs.
    """

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., description="Caller-supplied row id; echoed in response.")
    City: str = Field(..., max_length=64)
    State: str = Field(..., min_length=1, max_length=4, description="2-letter US state code.")
    Zip: int = Field(..., ge=0, le=99999)
    Bank: str = Field(..., max_length=128)
    BankState: str = Field(..., min_length=1, max_length=4)
    NAICS: int = Field(..., ge=0, description="6-digit NAICS; 0 is treated as missing.")
    NoEmp: int = Field(..., ge=0, description="Number of employees.")
    NewExist: float = Field(..., description="1.0 = existing, 2.0 = new; 0.0 treated as missing.")
    CreateJob: int = Field(..., ge=0)
    RetainedJob: int = Field(..., ge=0)
    FranchiseCode: int = Field(..., ge=0)
    UrbanRural: int = Field(..., ge=0, le=2, description="0 = undefined, 1 = urban, 2 = rural.")
    RevLineCr: str = Field(..., description="Y / N / other (mapped to U).")
    LowDoc: str = Field(..., description="Y / N / other (mapped to U).")
    DisbursementGross: float = Field(..., ge=0)
    BalanceGross: float = Field(..., ge=0)
    GrAppv: float = Field(..., gt=0, description="Approved loan amount; must be positive.")
    SBA_Appv: float = Field(..., ge=0, description="SBA-guaranteed portion.")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loans: list[LoanRequest] = Field(..., min_length=1, max_length=10_000)


class Prediction(BaseModel):
    index: int
    label: int = Field(..., description="0 = paid in full, 1 = predicted default.")
    probability_0: float = Field(..., ge=0.0, le=1.0)
    probability_1: float = Field(..., ge=0.0, le=1.0)


class FeatureDrift(BaseModel):
    psi: float
    status: str = Field(..., description="stable (<0.10), moderate (0.10-0.25) or shifted (>0.25)")


class ScoreDrift(BaseModel):
    psi: float
    status: str
    predicted_default_rate: float
    reference_default_rate: float | None = None


class DriftResponse(BaseModel):
    n_rows: int
    reference: dict
    features: dict[str, FeatureDrift]
    n_shifted: int
    n_moderate: int
    worst: list = Field(default_factory=list, description="five highest-PSI features")
    score_psi: ScoreDrift | None = None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    drift: DriftResponse | None = Field(None, description="present for batches of 200+ rows when a reference exists")


class ModelInfo(BaseModel):
    """What the model is, how it was trained, and the honest caveats.

    The ensemble-weight search selected model_1 alone (weights 1.0/0.0) — this
    is surfaced rather than marketed away. See README §"Honest disclosure".
    """

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    framework: str
    n_features: int
    feature_names: list[str]
    decision_threshold: float
    aucpr_test: float
    class_imbalance_scale_pos_weight: float
    ensemble_weights: dict[str, float]
    honest_disclosure: str
    held_out_evaluation: dict | None = Field(None, description="from scripts/evaluate.py on the reproduced test split")
    calibration: dict | None = None
    worst_slice_by_false_negative_rate: dict | None = None
    cost_weighted_threshold: dict | None = Field(None, description="threshold minimising the illustrative expected loss")
    training_data_vintage: str | None = None
    drift_reference: dict | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    version: str = ""
    drift_reference_loaded: bool = False
    uptime_s: float = 0.0
