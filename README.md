# SBA Loan Default Prediction

A production XGBoost classifier that scores U.S. Small Business Administration (SBA)
7(a) loan applications for default risk, shipped as a live Gradio app on Hugging Face
Spaces. Built around an explicit, audited eval pipeline (learning curves, residual
analysis, confusion matrices, ensemble weight search) and an interpretability layer
(SHAP bar/beeswarm/waterfall, permutation importance, surrogate decision tree) designed
for stakeholder review, not just a leaderboard score.

**Live app:** https://huggingface.co/spaces/bass990/SBA-Loan-Default-Prediction
**Author:** Mamadou Bassirou Diallo

---

## Headline results

| Metric | Test set | Notes |
|---|---|---|
| **AUCPR** | **0.5665** | Primary metric, appropriate under ~18% default rate (class imbalance) |
| Decision threshold | 0.66 | F1-optimal on the validation split, frozen before test scoring |
| Class imbalance handler | `scale_pos_weight = 4.71` | Computed from train-set negative/positive ratio |
| Engineered features | **31** | 16 raw + 15 derived (ratios, log transforms, flags, sector codes) |
| Best iteration | 1999 | Early-stopping on `aucpr` against a held-out validation set |

Held-out diagnostics are committed as PNGs under [artifacts/](artifacts/):
`confusion_matrices.png`, `learning_curves.png`, `residual_distributions.png`,
`ensemble_weight_search.png`, `shap_bar.png`, `shap_beeswarm.png`,
`shap_waterfall_scenarios.png`, `permutation_importance.png`, `surrogate_tree.png`.

---

## Problem framing

The SBA 7(a) program guarantees a portion of private bank loans to small businesses
that would otherwise be denied credit. The SBA absorbs the guaranteed share when a
loan defaults, so the loss cost is asymmetric: a false negative (approve, then
default) costs taxpayer-backed principal; a false positive (deny a loan that would
have repaid) costs a viable business its funding and the program its mission. A
single accuracy number obscures this asymmetry, so this project is evaluated on
**AUCPR** (precision-recall over the full operating curve) and a **business-aware
F1 threshold** rather than top-line accuracy.

**Modeled outcome.** Binary classification of `MIS_Status` ∈ {`P I F` paid in full,
`CHGOFF` charged off}, mapped to `0 / 1`.

**Operational frame.** The model's intended use is *decision support for a loan
officer*, not autonomous approval, the surrogate tree, SHAP plots, and probability
output are first-class outputs of the system, not afterthoughts.

---

## Approach

### Data
- Public SBA 7(a) loan-level dataset (`data/SBA_loans_project_2.zip`).
- A held-out validation file (`SBA_loans_project_2_holdout_students_valid.csv`, ~99.8K records) is used to exercise the scoring contract end-to-end.

### Feature engineering (31 features)
- **16 raw**: `State`, `Zip`, `BankState`, `NAICS`, `NoEmp`, `NewExist`, `CreateJob`, `RetainedJob`, `FranchiseCode`, `UrbanRural`, `RevLineCr`, `LowDoc`, `DisbursementGross`, `BalanceGross`, `GrAppv`, `SBA_Appv`.
- **15 derived** (stateless, same logic in notebook and scoring module):
  - **Ratios:** `sba_coverage_ratio` (SBA guarantee / requested), `disbursement_ratio` (disbursed / approved), `balance_ratio`, `sba_to_disbursement_ratio`, `jobs_per_employee`.
  - **Log transforms:** `log_disbursement`, `log_employees` (taming heavy right tails before tree splitting on monotonic transforms, XGBoost is invariant to monotonic transforms in theory but the log forms also stabilize quantile cuts during regularized tree growth).
  - **Flags:** `is_franchise`, `new_business`, `same_state_bank`, `rev_line_flag`, `low_doc_flag`, `naics_unknown`.
  - **Sector code:** `naics_sector` (NAICS // 10000), coarser than 6-digit NAICS, more samples per category, less overfit risk on rare industries.

### Modeling
- XGBoost binary classifier, `aucpr` eval metric, early stopping with patience.
- Optuna hyperparameter tuning across `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`.
- Class imbalance handled with `scale_pos_weight = 4.71` (computed from training-set ratio, not borrowed from theory).
- Two model variants trained with different sampling/regularization settings, then a **weighted-ensemble grid search** over weights `w₁ + w₂ = 1` selected the AUCPR-optimal combination on the validation set. See **Honest disclosure** below.

### Evaluation
- Train / validation / test split, frozen before tuning.
- AUCPR as the primary metric, with confusion matrices at the chosen F1 threshold for the operational picture.
- Learning curves to diagnose under-/over-fitting at the chosen capacity.
- Residual distributions over predicted probability, visualizes calibration behavior across the score range.

### Interpretability layer
Stakeholders for an SBA-style model are loan officers and policy reviewers, not ML engineers. The interpretability layer is designed for them, not for me:
- **SHAP bar:** top global drivers.
- **SHAP beeswarm:** direction + magnitude per feature, distributional.
- **SHAP waterfall:** per-application case studies (`shap_waterfall_scenarios.png`).
- **Permutation importance:** model-agnostic sanity check vs. SHAP, when they disagree, it's a flag.
- **Surrogate decision tree:** a shallow, human-readable approximation (`surrogate_tree.png`), what the policy team can actually argue about.

---

## Honest disclosure (the part that matters)

**Two models were trained; the ensemble weight search selected model 1 alone.**

The `ensemble_weight_search.png` plot sweeps `(w₁, w₂)` and reports validation AUCPR
at each weight pair. The optimum at `(1.0, 0.0)` is preserved verbatim in
`artifacts/ensemble_metadata.json`. The shipped model is **model 1 only**, with the
weight-search ablation documented transparently:

| Model | Validation AUCPR | Ensemble weight |
|---|---|---|
| Model 1 | **0.5665** | **1.0** (shipped) |
| Model 2 | 0.5419 | 0.0 |

This is not a hidden detail being polished out of the marketing copy, it is the
methodology. Running the ensemble search and reporting the outcome honestly is the
work; pretending the "ensemble" produced a lift it didn't would be a worse system.
In a hiring loop, expect me to walk through both training configs and explain why
model 2 underperformed (different `scale_pos_weight` × `min_child_weight` regime , 
it traded recall for precision in a region the F1 threshold wasn't going to favor).

---

## Repo structure

```
sba-loan-default-prediction/
├── README.md                                           ← you are here
├── problem_brief.md                                    ← problem spec, dataset, modeling constraints, scoring contract
├── requirements.txt                                    ← pinned, Python 3.12.10
├── sba_loan_default_model.ipynb                        ← end-to-end training notebook
├── score_model.py                                      ← production scoring entrypoint
├── artifacts/                                          ← models + encoders + diagnostic plots
│   ├── xgb_model_1.json                                ← shipped model (weight 1.0)
│   ├── xgb_model_2.json                                ← weight 0.0, kept for audit
│   ├── ensemble_metadata.json                          ← weights, threshold, feature order
│   ├── label_encoders.pkl                              ← fit on train, applied at score
│   ├── median_fill.pkl                                 ← per-feature train-set medians
│   └── *.png                                           ← diagnostic plots
├── data/                                               ← raw + holdout schema file (gitignored)
└── huggingface-spaces/                                 ← Gradio app + deploy notes
    ├── app.py
    ├── requirements.txt
    └── deployment_notes.md
```

---

## Reproduce locally

```bash
# Python 3.12.10 required
uv venv --python 3.12.10 .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate           # macOS/Linux
uv pip sync requirements.txt
python -m ipykernel install --user --name sba-loan-default --display-name "SBA loan default"

# Verify the scoring contract on the holdout file:
python score_model.py
# → "Scoring function validation passed." and a 5-row preview
```

To re-train end-to-end, open the notebook in the `SBA loan default` kernel and run
top to bottom. Saved artifacts will be overwritten in `artifacts/`.

---

## Scoring contract

`score(input_df)` returns a 4-column DataFrame:

| Column | Type | Meaning |
|---|---|---|
| `index` | int | Pass-through of the input record ID, order preserved |
| `label` | int (0 or 1) | Predicted class at the F1-optimal threshold (0.66) |
| `probability_0` | float | Ensemble probability of paid-in-full |
| `probability_1` | float | Ensemble probability of default, `probability_0 + probability_1 == 1.0` |

The scoring module resolves artifact paths relative to its own file so a serving
wrapper can import it from any working directory. Output is validated before return,
schema mismatch raises rather than silently producing bad scores.

---

## Live app

The Hugging Face Space wraps the same scoring function in a Gradio UI:

1. Upload a CSV in the project schema (same columns as training, no `MIS_Status`).
2. Click **Score**.
3. The right panel shows: record count, predicted default rate, ensemble weights, threshold.
4. The results table renders the 4-column scoring output.
5. **Download Results CSV** persists the scored rows.

Screenshots live in [huggingface-spaces/](huggingface-spaces/). Deployment steps are
in `huggingface-spaces/deployment_notes.md`.

---

## What I'd want before deploying this for real

This is a portfolio model, not an SBA-approved underwriting model. If a loan officer
were going to use it on a live application tomorrow, I would want:

1. **Drift monitoring.** Training data ends in 2014. NAICS code distributions, the
   `LowDoc` program rules, and the SBA guarantee fraction policy have all shifted
   since. A monthly PSI on the top-10 SHAP features against the live application
   stream is the minimum monitoring posture.
2. **Slice analysis.** AUCPR is a global number. Before signing off, I'd want
   per-state, per-NAICS-sector, per-loan-size-bucket AUCPR and false-negative-rate
   tables, with explicit minima the model must clear to stay in service. The
   current notebook stops at the global figure; the slice cut is a Phase 2
   project upgrade in my plan.
3. **Cost-weighted threshold.** F1 weighs precision and recall equally. The real
   loss is `E[loss] = P(default) × LGD × exposure − P(repay) × interest_margin`.
   Replacing the F1 threshold with an `argmax E[$]` threshold over the validation
   set is a one-day fix that changes nothing structural and changes the operating
   point materially.
4. **Calibration.** Tree ensembles produce probabilities that are usefully
   *ranked* but not always usefully *scaled*. For business decisions that read the
   probability number itself (e.g., expected-loss calculations), I'd want
   isotonic or Platt calibration on the validation set with a reliability-diagram
   check.
5. **Adverse-action transparency.** ECOA requires lenders to give applicants
   specific reasons for denial. The SHAP waterfall plots are a starting point, but
   the production system would need a curated reason-code dictionary that maps
   SHAP attributions to legally compliant language, not the raw feature names.

---

## Failure modes (where I expect this model to break)

- **Vintage mismatch.** The public SBA 7(a) dataset ends well before today's
  lending environment. Any post-training economic shock regime (COVID PPP overlap,
  post-2022 rate environment, regional banking stress) is unrepresented.
- **`naics_unknown` flag.** Roughly tracks SBA data-entry quality more than risk.
  In a clean live system this flag would be near-zero, removing a signal the model
  has learned to lean on.
- **Geographic concentration.** A handful of states dominate volume. Rare-state
  scoring is more an `out-of-distribution` extrapolation than a prediction; the
  label encoder maps unseen states to `UNSEEN` and the median imputer catches the
  rest, but neither addresses the underlying small-sample problem.
- **`LowDoc` and `RevLineCr` semantics.** Both fields are dirty in the raw data
  (`Y` / `N` / blank / `0` / `T` all appear). The cleaning rule maps anything
  unrecognized to `U`. If a downstream consumer changes those codes' meaning, the
  feature silently drifts.

---

## Tech stack

Python 3.12.10 · xgboost 3.1.2 · scikit-learn 1.8.0 · optuna 4.6.0 · shap 0.51.0 ·
pandas 2.3.3 · numpy 2.0.2 · matplotlib 3.10.8 · seaborn 0.13.2 · gradio 6.13.0 ·
category-encoders 2.9.0. Full pinned set in [requirements.txt](requirements.txt).

---

## License

MIT (project code). The SBA 7(a) dataset is a U.S. government public-records release.

---

## About this project

This project started as a graduate Machine Learning coursework assignment at UT Dallas in early 2026. The dataset, the broad modeling constraint (two-XGBoost ensemble, AUCPR optimization, scoring-contract requirements), and the deployment requirement came from that assignment brief, preserved as `problem_brief.md`. Everything else, the feature engineering decisions, the ensemble weight search and the honest-disclosure framing of the result, the SHAP and surrogate-tree interpretability layer, the operational framing around AUCPR, the production scoring contract, the failure-mode analysis, and the Hugging Face deployment, is my work. I extended it for portfolio with the interpretability layer and the "what I'd want before deploying for real" section.
