# Problem brief, SBA loan default prediction

A binary classification problem on the U.S. Small Business Administration (SBA) loan-default dataset. The goal is to predict whether a small-business loan will default (a `MIS_Status == 1` outcome) given the borrower, loan, and economic-context features available at origination.

This document captures the technical constraints I worked under. The model build and scoring contract in this repo were designed to satisfy them.

## Problem

For each loan record, predict the probability that the loan ends in default. The deliverable is twofold:

1. A trained classification model with a tuned decision threshold, evaluated on a held-out test split, with diagnostic plots and feature attribution.
2. A standalone scoring function (`score_model.py`) that takes a DataFrame of unseen records in the original input schema and returns a four-column scoring output. The scoring contract has to be robust to records that have never been seen by the model.

## Dataset

- Training data: `data/SBA_loans_project_2.zip` (loaded by the notebook).
- Schema-validation file: `data/SBA_loans_project_2_holdout_students_valid.csv` (~99.8K records, unlabeled, used to verify the scoring function processes unseen-format input correctly).

The training target is `MIS_Status`. Positive class (`MIS_Status == 1`) is loan default. Negative class (`MIS_Status == 0`) is non-default.

The dataset includes an `index` column. It is preserved as the per-record identifier in scoring output but is never used as a model feature.

The holdout file is for schema and input-handling validation only. It is not a validation set for modeling decisions. No model selection, hyperparameter tuning, threshold optimization, or test metrics use it.

## Modeling constraints

These are the design constraints I followed:

- **Two distinct XGBoost models**, trained with the native XGBoost API (not the scikit-learn wrapper), combined via a weighted-average ensemble with the weights tuned on a validation split.
- **AUCPR as the primary tuning metric**, on the grounds that the loan-default base rate is imbalanced and AUCPR is more informative than ROC AUC at the low-prevalence end.
- **Decision threshold optimized for F1 on the validation split**, then frozen before any test scoring.
- **Leakage-safe preprocessing.** All target encoders, weight-of-evidence transforms, scalers, and median-fill imputers are fit on the training split only and applied unchanged at scoring time.
- **SHAP feature attribution** on one of the two XGBoost component models, plus permutation importance, individual-prediction explanations, and residual analysis.

## Scoring contract

The scoring function `score(input_df)` is the production interface. The contract:

**Inputs:** a pandas DataFrame in the original SBA loan schema (no `MIS_Status` column required).

**Output:** a pandas DataFrame with exactly four columns: the original `index`, the predicted probability of default, the predicted binary label at the frozen threshold, and a confidence band. Same row count as input. Same row order as input.

**Internal behavior the scoring function must obey:**

- Do not drop records, even if they have unseen categorical values or missing fields.
- Do not refit any encoder, scaler, imputer, feature selector, or model. All transforms are loaded from `artifacts/` and applied as-fit.
- Do not depend on notebook state or any in-memory variable that wasn't loaded from disk.
- Resolve artifact paths relative to the scoring file's own location, so the function works when imported from a different working directory (which is what the deployed Gradio app does).

These rules are the difference between a scoring function that works in a notebook and one that survives unseen production input. The most common failure mode in this kind of project is "the function was never tested outside the notebook before submission."

## Reproducibility

- Python 3.12.10.
- Package versions pinned in `requirements.txt`. The notebook and scoring script are tested against those exact versions.
- Random seeds fixed where applicable.
- Relative paths from the repo root in notebooks. Self-resolving paths in the scoring script.
- No secrets, tokens, API keys, or private credentials in the repo.

## Deployment

The same scoring function is wrapped in a Gradio UI and deployed to Hugging Face Spaces. Deployment notes and screenshots live under `huggingface-spaces/`.
