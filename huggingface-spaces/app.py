"""
SBA Loan Default Prediction — Gradio Scoring App
Hugging Face Spaces deployment for Project 2 (BUAN 6341, Spring 2026)
Author: Mamadou Bassirou Diallo  
"""

import io
import json
import pickle
import traceback
from pathlib import Path
import tempfile, os


import gradio as gr
import numpy as np
import pandas as pd
import xgboost as xgb

# ── Artifact paths ─────────────────────────────────────────────────────────────
HERE          = Path(__file__).resolve().parent
ARTIFACTS_DIR = HERE / "artifacts"
if not ARTIFACTS_DIR.exists():
    ARTIFACTS_DIR = HERE.parent / "artifacts"   # fallback for local dev


# ── Artifact loaders ───────────────────────────────────────────────────────────
def _load_booster(path: Path) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(path))
    return b


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_json(path: Path):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with path.open("r", encoding=enc) as fh:
                return json.load(fh)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path}")


# ── Cleaning helpers ───────────────────────────────────────────────────────────
def _clean_revlinecr(s: pd.Series) -> pd.Series:
    s = s.copy().astype(str).str.strip().str.upper()
    return s.map(lambda x: x if x in ("Y", "N") else "U")


def _clean_lowdoc(s: pd.Series) -> pd.Series:
    s = s.copy().astype(str).str.strip().str.upper()
    return s.map(lambda x: x if x in ("Y", "N") else "U")


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RevLineCr"] = _clean_revlinecr(out["RevLineCr"])
    out["LowDoc"]    = _clean_lowdoc(out["LowDoc"])
    out["NewExist"]  = out["NewExist"].where(out["NewExist"] != 0.0, other=np.nan)
    naics = out["NAICS"].astype(float)
    out["NAICS"]     = naics.where(naics != 0, other=np.nan)
    out["City"]      = out["City"].fillna("UNKNOWN")
    out["State"]     = out["State"].fillna("UNK")
    out["Bank"]      = out["Bank"].fillna("UNKNOWN")
    out["BankState"] = out["BankState"].fillna("UNK")
    return out


# ── Feature engineering ────────────────────────────────────────────────────────
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


# ── Preprocessing ──────────────────────────────────────────────────────────────
def _preprocess(df, metadata, label_encoders, median_fill):
    out = df.copy()
    drop_cols        = metadata["drop_cols"]
    categorical_cols = metadata["categorical_cols"]
    newexist_mode    = metadata["newexist_mode"]
    feature_names    = metadata["feature_names"]

    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    out = out.drop(columns=["MIS_Status"], errors="ignore")

    out["NewExist"] = out["NewExist"].fillna(newexist_mode)

    for col in categorical_cols:
        le   = label_encoders[col]
        vals = out[col].fillna("MISSING").astype(str)
        vals = vals.map(lambda v, le=le: v if v in le.classes_ else "UNSEEN")
        out[col] = le.transform(vals)

    for col in feature_names:
        if col not in out.columns:
            out[col] = median_fill.get(col, 0.0)

    out = out[feature_names]

    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(median_fill.get(col, 0.0))

    return out


# ── Lazy-load artifacts once ───────────────────────────────────────────────────
_artifacts = {}

def _get_artifacts():
    if _artifacts:
        return _artifacts
    _artifacts["model_1"]        = _load_booster(ARTIFACTS_DIR / "xgb_model_1.json")
    _artifacts["model_2"]        = _load_booster(ARTIFACTS_DIR / "xgb_model_2.json")
    _artifacts["label_encoders"] = _load_pickle(ARTIFACTS_DIR / "label_encoders.pkl")
    _artifacts["median_fill"]    = _load_pickle(ARTIFACTS_DIR / "median_fill.pkl")
    _artifacts["metadata"]       = _load_json(ARTIFACTS_DIR / "ensemble_metadata.json")
    return _artifacts


# ── Main prediction function ───────────────────────────────────────────────────
def predict_csv(file_obj):
    if file_obj is None:
        return None, "Please upload a CSV file."

    # ── Read CSV — try multiple encodings ─────────────────────────────────────
    input_df = None
    for enc in ("latin-1", "utf-8", "cp1252"):
        try:
            input_df = pd.read_csv(file_obj.name, encoding=enc)
            break
        except Exception:
            continue

    if input_df is None:
        try:
            input_df = pd.read_csv(
                io.StringIO(open(file_obj.name, "rb").read().decode("latin-1"))
            )
        except Exception as e:
            return None, f"Could not read CSV: {e}"

    if "index" not in input_df.columns:
        return None, "CSV must contain an 'index' column."

    # ── Score ──────────────────────────────────────────────────────────────────
    try:
        arts        = _get_artifacts()
        model_1     = arts["model_1"]
        model_2     = arts["model_2"]
        le          = arts["label_encoders"]
        med         = arts["median_fill"]
        meta        = arts["metadata"]

        weight_m1   = meta["weight_model_1"]
        weight_m2   = meta["weight_model_2"]
        threshold   = meta["threshold_ensemble"]
        feat_names  = meta["feature_names"]
        best_iter_1 = meta["model_1_best_iteration"]
        best_iter_2 = meta["model_2_best_iteration"]

        record_ids = input_df["index"].reset_index(drop=True)

        data = _clean_df(input_df.copy())
        data = _engineer_features(data)
        data = _preprocess(data, meta, le, med)

        dmat_1 = xgb.DMatrix(data, feature_names=feat_names)
        dmat_2 = xgb.DMatrix(data, feature_names=feat_names)

        prob1_m1 = model_1.predict(dmat_1, iteration_range=(0, best_iter_1 + 1))
        prob1_m2 = model_2.predict(dmat_2, iteration_range=(0, best_iter_2 + 1))

        probability_1 = weight_m1 * prob1_m1 + weight_m2 * prob1_m2
        probability_0 = 1.0 - probability_1
        labels        = (probability_1 >= threshold).astype(int)

        results = pd.DataFrame({
            "index":         record_ids,
            "label":         labels,
            "probability_0": probability_0.round(6),
            "probability_1": probability_1.round(6),
        })

        n_default = int((results["label"] == 1).sum())
        pct       = 100 * n_default / len(results)
        summary   = (
            f"Scored {len(results):,} records successfully.\n"
            f"Predicted defaults (label=1): {n_default:,} ({pct:.1f}%)\n"
            f"Ensemble weights: w1={weight_m1:.2f}, w2={weight_m2:.2f}\n"
            f"Decision threshold: {threshold:.4f}"
        )
        return results, summary

    except Exception:
        return None, f"Scoring error:\n{traceback.format_exc()}"


# ── Gradio interface ───────────────────────────────────────────────────────────
with gr.Blocks(title="SBA Loan Default Prediction") as demo:
    gr.Markdown(
        """
        # SBA Loan Default Prediction

        Assess default risk on SBA loan portfolios instantly. Upload a CSV of loan records
        and receive probability scores and labels for every record — powered by a tuned XGBoost ensemble.

        *Mamadou Bassirou Diallo*
        """
    )

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])

    score_btn = gr.Button("Score", variant="primary")

    with gr.Row():
        status_box = gr.Textbox(label="Status / Summary", lines=8)

    results_table = gr.Dataframe(
        label="Scoring Results (index | label | probability_0 | probability_1)",
        interactive=False,
    )

    download_btn = gr.DownloadButton(label="Download Results CSV")

    def run_and_prepare_download(file_obj):
        results_df, summary = predict_csv(file_obj)
        if results_df is None:
            return summary, None, gr.update(visible=False)
        tmp_path = os.path.join(tempfile.gettempdir(), "scored_output.csv")
        results_df.to_csv(tmp_path, index=False)
        return summary, results_df, gr.update(value=tmp_path, visible=True)

    score_btn.click(
        run_and_prepare_download,
        inputs=[file_input],
        outputs=[status_box, results_table, download_btn],
    )

if __name__ == "__main__":
    demo.launch()
