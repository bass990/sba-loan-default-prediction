"""
SBA Loan Default Prediction, Gradio scoring app (Hugging Face Space).

Scoring is delegated to `score_model.py`, the same module the FastAPI service
and the golden-prediction tests use, so the Space cannot drift from the API.
Deploy: upload app.py, score_model.py, requirements.txt, sample_500.csv and the
artifacts/ directory (see deployment_notes.md).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):           # Space root, or the repo when run locally
    if (candidate / "score_model.py").exists():
        sys.path.insert(0, str(candidate))
        break
from score_model import load_artifacts, score  # noqa: E402

SAMPLE_CSV = HERE / "sample_500.csv"
EVAL_REPORT = next((p for p in (HERE / "artifacts" / "eval_report.json", HERE.parent / "artifacts" / "eval_report.json") if p.exists()), None)
MODEL_CARD_URL = "https://github.com/bass990/sba-loan-default-prediction#model-card"


def _eval_numbers() -> dict:
    """Headline + threshold findings from the evaluation script, so the UI states what the repo measured."""
    out = {"aucpr": 0.567, "ece": 0.21, "f1_threshold": 0.66, "cost_threshold": 0.35, "loss_f1": None, "loss_cost": None}
    if EVAL_REPORT:
        try:
            r = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
            out["aucpr"] = r["headline"]["aucpr"]
            out["ece"] = r["calibration"]["ece"]
            cw = r.get("cost_weighted_threshold", {})
            out["cost_threshold"] = cw.get("best_threshold", {}).get("threshold", 0.35)
            out["loss_cost"] = cw.get("best_threshold", {}).get("expected_loss_per_loan")
            out["loss_f1"] = cw.get("f1_threshold", {}).get("expected_loss_per_loan")
        except (OSError, ValueError, KeyError):
            pass
    return out


EVAL = _eval_numbers()
THRESHOLDS = {
    f"F1-optimal, {EVAL['f1_threshold']:.2f} (model card)": EVAL["f1_threshold"],
    f"Cost-optimal, {EVAL['cost_threshold']:.2f} (lowest expected loss per loan)": EVAL["cost_threshold"],
}


def _read_csv(path: str) -> pd.DataFrame:
    for enc in ("latin-1", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("could not decode the CSV (tried latin-1, utf-8, cp1252)")


def predict_csv(file_obj, threshold_label: str):
    if file_obj is None:
        return "Upload a CSV or load the sample first.", None, gr.update(visible=False)
    threshold = THRESHOLDS.get(threshold_label, EVAL["f1_threshold"])
    try:
        input_df = _read_csv(file_obj.name)
        if "index" not in input_df.columns:
            return "The CSV needs an 'index' column (record id).", None, gr.update(visible=False)
        scored = score(input_df)                                  # shared module: cleaning, features, ensemble
        prob = scored["probability_1"].astype(float)
        results = pd.DataFrame({
            "index": scored["index"],
            "predicted_default": (prob >= threshold).astype(int),
            "default_probability": prob.round(3),
        })
        n = len(results)
        n_default = int(results["predicted_default"].sum())
        loss_note = ""
        if EVAL["loss_cost"] and EVAL["loss_f1"]:
            loss_note = (f"On the held-out test set the cost-optimal threshold lowers expected loss per loan from "
                         f"${EVAL['loss_f1']:,.0f} to ${EVAL['loss_cost']:,.0f} (approves fewer loans, catches more defaults).\n")
        summary = (
            f"Scored {n:,} records with the XGBoost model (held-out AUCPR {EVAL['aucpr']:.3f}; base rate about 18%).\n"
            f"Threshold {threshold:.2f}: {n_default:,} flagged as likely default ({100 * n_default / n:.1f}%).\n"
            + loss_note +
            f"Calibration warning: the probabilities are not calibrated (ECE {EVAL['ece']:.2f}); they over-state default risk "
            f"roughly two to three times in the mid range. Use them to rank and threshold, not as literal default chances.\n"
            f"Model card and evaluation: {MODEL_CARD_URL}"
        )
        tmp = os.path.join(tempfile.gettempdir(), "sba_scored.csv")
        results.to_csv(tmp, index=False)
        return summary, results, gr.update(value=tmp, visible=True)
    except Exception:  # noqa: BLE001 - surface the traceback to the user of a scoring UI
        return f"Scoring error:\n{traceback.format_exc()}", None, gr.update(visible=False)


def load_sample():
    return str(SAMPLE_CSV) if SAMPLE_CSV.exists() else None


with gr.Blocks(title="SBA Loan Default Prediction") as demo:
    gr.Markdown(
        f"""
        # SBA Loan Default Prediction

        Upload a CSV of SBA loan records (same columns as the SBA Project 2 dataset) and get a default flag and
        probability for every record from a tuned XGBoost model. Held-out AUCPR **{EVAL['aucpr']:.3f}** against a
        base rate of ~18%; the probabilities are **not calibrated** (ECE {EVAL['ece']:.2f}), so treat them as a
        ranking score. Full model card, slices and the threshold analysis: [{MODEL_CARD_URL}]({MODEL_CARD_URL}).

        *Mamadou Bassirou Diallo*
        """
    )
    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        with gr.Column():
            threshold_choice = gr.Radio(list(THRESHOLDS), value=list(THRESHOLDS)[0], label="Decision threshold",
                                        info="0.66 maximises F1 (the model card number); 0.35 minimises expected dollar loss under the repo's cost model.")
            sample_btn = gr.Button("Load 500-row sample", variant="secondary")
    score_btn = gr.Button("Score", variant="primary")
    status_box = gr.Textbox(label="Summary", lines=7)
    results_table = gr.Dataframe(label="Results (index | predicted_default | default_probability)", interactive=False)
    download_btn = gr.DownloadButton(label="Download results CSV", visible=False)

    sample_btn.click(load_sample, inputs=None, outputs=[file_input])
    score_btn.click(predict_csv, inputs=[file_input, threshold_choice], outputs=[status_box, results_table, download_btn])

if __name__ == "__main__":
    load_artifacts()   # fail fast if the artifacts are missing
    demo.launch()
