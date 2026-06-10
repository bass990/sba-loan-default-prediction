"""
Project 2 Scoring Function
SBA Loan Default Prediction
Author: Mamadou Bassirou Diallo 


Entry point: score(input_df) -> pd.DataFrame
Artifact paths are resolved relative to THIS file so the grader can import
from any working directory.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

# ── Path resolution ─────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR      = PROJECT_ROOT / "data"
HOLDOUT_FORMAT_PATH = DATA_DIR / "SBA_loans_project_2_holdout_students_valid.csv"

REQUIRED_OUTPUT_COLUMNS = ["index", "label", "probability_0", "probability_1"]


# ── Artifact loaders ────────────────────────────────────────────────────────
def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_booster(path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


# ── Output validation ────────────────────────────────────────────────────────
def _validate_scoring_output(predictions: pd.DataFrame, input_df: pd.DataFrame) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("score() must return a pandas DataFrame")
    if list(predictions.columns) != REQUIRED_OUTPUT_COLUMNS:
        raise ValueError(f"Output columns must be exactly {REQUIRED_OUTPUT_COLUMNS}")
    if len(predictions) != len(input_df):
        raise ValueError("Output row count must match input row count")
    if not predictions["label"].isin([0, 1]).all():
        raise ValueError("Predicted labels must be 0 or 1")
    prob_sum = predictions["probability_0"] + predictions["probability_1"]
    if not np.allclose(prob_sum, 1.0, atol=1e-6):
        raise ValueError("probability_0 and probability_1 must sum to 1")
    if "index" in input_df.columns and not predictions["index"].equals(
        input_df["index"].reset_index(drop=True)
    ):
        raise ValueError("Output index column must preserve the input index values and order")


# ── Cleaning helpers (stateless — same logic as notebook) ───────────────────
def _clean_revlinecr(series: pd.Series) -> pd.Series:
    s = series.copy().astype(str).str.strip().str.upper()
    return s.map(lambda x: x if x in ("Y", "N") else "U")


def _clean_lowdoc(series: pd.Series) -> pd.Series:
    s = series.copy().astype(str).str.strip().str.upper()
    return s.map(lambda x: x if x in ("Y", "N") else "U")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RevLineCr"] = _clean_revlinecr(out["RevLineCr"])
    out["LowDoc"]    = _clean_lowdoc(out["LowDoc"])
    # NewExist: replace 0.0 with NaN (invalid code)
    out["NewExist"]  = out["NewExist"].where(out["NewExist"] != 0.0, other=np.nan)
    # NAICS: replace 0 with NaN
    naics = out["NAICS"].astype(float)
    out["NAICS"] = naics.where(naics != 0, other=np.nan)
    # Fill simple string NaNs
    out["City"]      = out["City"].fillna("UNKNOWN")
    out["State"]     = out["State"].fillna("UNK")
    out["Bank"]      = out["Bank"].fillna("UNKNOWN")
    out["BankState"] = out["BankState"].fillna("UNK")
    return out


# ── Feature engineering (stateless — same logic as notebook) ────────────────
def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sba_coverage_ratio"]        = np.where(out["GrAppv"] > 0, out["SBA_Appv"] / out["GrAppv"], 0.0)
    out["disbursement_ratio"]        = np.where(out["GrAppv"] > 0, out["DisbursementGross"] / out["GrAppv"], 0.0)
    out["balance_ratio"]             = out["BalanceGross"] / (out["DisbursementGross"] + 1.0)
    out["is_franchise"]              = (out["FranchiseCode"] > 1).astype(int)
    out["naics_unknown"]             = out["NAICS"].isna().astype(int)
    out["naics_sector"]              = (out["NAICS"].fillna(0) // 10000).astype(int)
    out["total_jobs"]                = out["CreateJob"] + out["RetainedJob"]
    out["jobs_per_employee"]         = out["total_jobs"] / (out["NoEmp"] + 1.0)
    out["new_business"]              = (out["NewExist"].fillna(1.0) == 2.0).astype(int)
    out["same_state_bank"]           = (out["State"] == out["BankState"]).astype(int)
    out["log_disbursement"]          = np.log1p(out["DisbursementGross"])
    out["log_employees"]             = np.log1p(out["NoEmp"])
    out["sba_to_disbursement_ratio"] = out["SBA_Appv"] / (out["DisbursementGross"] + 1.0)
    out["rev_line_flag"]             = (out["RevLineCr"] == "Y").astype(int)
    out["low_doc_flag"]              = (out["LowDoc"]    == "Y").astype(int)
    return out


# ── Preprocessing (uses saved encoders/stats — no fitting) ──────────────────
def _preprocess(df: pd.DataFrame, metadata: dict,
                label_encoders: dict, median_fill: dict) -> pd.DataFrame:
    out = df.copy()
    drop_cols        = metadata["drop_cols"]
    categorical_cols = metadata["categorical_cols"]
    newexist_mode    = metadata["newexist_mode"]
    feature_names    = metadata["feature_names"]

    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    out = out.drop(columns=["MIS_Status"], errors="ignore")

    # Impute NewExist
    out["NewExist"] = out["NewExist"].fillna(newexist_mode)

    # Label encode categoricals
    for col in categorical_cols:
        le   = label_encoders[col]
        vals = out[col].fillna("MISSING").astype(str)
        vals = vals.map(lambda v, le=le: v if v in le.classes_ else "UNSEEN")
        out[col] = le.transform(vals)

    # Reorder / add missing columns to match training feature set
    for col in feature_names:
        if col not in out.columns:
            out[col] = median_fill.get(col, 0.0)

    out = out[feature_names]

    # Fill any remaining NaN with training medians
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(median_fill.get(col, 0.0))

    return out


# ── Main scoring entry point ─────────────────────────────────────────────────
def score(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score Project 2 input records using the final two-model XGBoost ensemble.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input data in the same format as Project 2 training data,
        without the MIS_Status column.

    Returns
    -------
    pd.DataFrame
        Columns: index, label, probability_0, probability_1
    """
    if "index" not in input_df.columns:
        raise ValueError("Input data must include the 'index' column")

    data       = input_df.copy()
    record_ids = data["index"].reset_index(drop=True)

    # Load artifacts
    model_1       = _load_booster(ARTIFACTS_DIR / "xgb_model_1.json")
    model_2       = _load_booster(ARTIFACTS_DIR / "xgb_model_2.json")
    label_encoders = _load_pickle(ARTIFACTS_DIR / "label_encoders.pkl")
    median_fill    = _load_pickle(ARTIFACTS_DIR / "median_fill.pkl")
    metadata       = _load_json(ARTIFACTS_DIR / "ensemble_metadata.json")

    weight_m1  = metadata["weight_model_1"]
    weight_m2  = metadata["weight_model_2"]
    threshold  = metadata["threshold_ensemble"]
    feature_names = metadata["feature_names"]
    best_iter_m1  = metadata["model_1_best_iteration"]
    best_iter_m2  = metadata["model_2_best_iteration"]

    # Clean → feature engineer → preprocess
    data = _clean_df(data)
    data = _engineer_features(data)
    data = _preprocess(data, metadata, label_encoders, median_fill)

    # Build DMatrices
    dmat_1 = xgb.DMatrix(data, feature_names=feature_names)
    dmat_2 = xgb.DMatrix(data, feature_names=feature_names)

    # Predict
    prob1_m1 = model_1.predict(dmat_1, iteration_range=(0, best_iter_m1 + 1))
    prob1_m2 = model_2.predict(dmat_2, iteration_range=(0, best_iter_m2 + 1))

    # Ensemble
    probability_1 = weight_m1 * prob1_m1 + weight_m2 * prob1_m2
    probability_0 = 1.0 - probability_1
    labels        = (probability_1 >= threshold).astype(int)

    predictions = pd.DataFrame({
        "index":         record_ids,
        "label":         labels,
        "probability_0": probability_0,
        "probability_1": probability_1,
    })

    _validate_scoring_output(predictions, input_df)
    return predictions


# ── Local validation helper ───────────────────────────────────────────────────
def validate_scoring_function() -> pd.DataFrame:
    """Run a local schema check using the holdout-format file."""
    if not HOLDOUT_FORMAT_PATH.exists():
        raise FileNotFoundError(f"Holdout file not found: {HOLDOUT_FORMAT_PATH}")
    holdout_df  = pd.read_csv(HOLDOUT_FORMAT_PATH)
    predictions = score(holdout_df)
    _validate_scoring_output(predictions, holdout_df)
    print("Scoring function validation passed.")
    return predictions


if __name__ == "__main__":
    result = validate_scoring_function()
    print(result.head())
