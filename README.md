# SBA Loan Default Prediction

![The scoring app after scoring the 500-row sample at the F1-optimal threshold: the summary box reports AUCPR 0.567, 89 of 500 flagged, the expected-loss comparison between the two thresholds and the calibration warning, followed by the results table](./docs/screenshots/scoring_app.png)

*The same Gradio app the Hugging Face Space serves, here from the Docker image at `/ui`. The threshold radio exposes the cost-optimal 0.35 next to the model-card 0.66, and the calibration warning is on the page, not in a footnote.*

A production XGBoost classifier that scores U.S. Small Business Administration (SBA)
7(a) loan applications for default risk, shipped as a live Gradio app on Hugging Face
Spaces. Built around an explicit, audited eval pipeline (learning curves, residual
analysis, confusion matrices, ensemble weight search) and an interpretability layer
(SHAP bar/beeswarm/waterfall, permutation importance, surrogate decision tree) designed
for stakeholder review, not just a leaderboard score. Version 1.1 adds what a reviewer
asks for next: a held-out evaluation that reproduces the headline and then measures
calibration, per-segment false-negative rates and a cost-weighted operating point; a
FastAPI service with PSI drift monitoring, golden-prediction regression tests and a
Docker image with a CI smoke test.

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

## Held-out evaluation: calibration, slices, cost-weighted threshold

`scripts/evaluate.py` reproduces the notebook's stratified 15% test split (seed 42,
119,904 loans) from the labelled archive, scores it with the production
`score_model.score()`, and writes [`artifacts/eval_report.json`](artifacts/eval_report.json).
The API serves the same numbers on `/model-info`. Nothing is refit.

**Headline reproduces.** AUCPR 0.5666 on the re-created split vs 0.5665 on the model
card; ROC AUC 0.838; at the shipped 0.66 threshold precision 0.527, recall 0.551, F1 0.538.

**The probabilities are well ranked and badly scaled.** `scale_pos_weight = 4.71` does
what it is meant to for ranking, and it also inflates every probability. Expected
calibration error is **0.206**, Brier 0.160:

| Predicted default probability | Loans | Observed default rate |
|---|---|---|
| 0.1-0.2 | 20,229 | 3.7% |
| 0.3-0.4 | 14,690 | 10.2% |
| 0.5-0.6 | 10,640 | 22.2% |
| 0.7-0.8 | 7,916 | 41.7% |
| 0.9-1.0 | 3,608 | 81.2% |

A loan the model calls "60% likely to default" defaults about 30% of the time. Anyone
reading the number as a probability (expected-loss maths, pricing) needs the
isotonic/Platt step that is still on the roadmap; anyone reading it as a rank does not.

**The model is weakest where the money is.** False-negative rate at the 0.66 threshold
by loan size:

| Loan size (`GrAppv`) | Loans | Default rate | AUCPR | False-negative rate |
|---|---|---|---|---|
| ≤ $50k | 46,650 | 26.4% | 0.631 | 30% |
| $50-150k | 34,106 | 14.7% | 0.491 | 61% |
| $150-500k | 27,604 | 9.8% | 0.429 | 71% |
| > $500k | 11,544 | 8.5% | 0.343 | **83%** |

The F1-optimal threshold is tuned on the small-loan-dominated population; on loans
above $500k it misses five defaults in six. By state, AUCPR ranges from 0.69 (FL) to
0.39 (MN, PA) among the ten largest states. These are the slice tables the earlier
README said it would want before deployment; now they exist and they argue for
per-segment thresholds.

**The threshold is a business decision, not an F1 decision.** Under an illustrative
loss model (loss-given-default 50% of the SBA-guaranteed amount, 5% margin on a loan
that repays; both are parameters of `evaluate.py`, not measured SBA economics), the
expected loss per loan is minimised at a threshold of **0.35** ($4,140 per loan,
approving 51% of applications, catching 88% of defaults), against **$5,330** at the
F1 threshold of 0.66 (approving 81%, catching 56%). Same model, same probabilities,
a 22% lower expected loss from moving the operating point, which is why the F1
threshold ships with the loss sweep next to it rather than alone.

---

## Repo structure

```
sba-loan-default-prediction/
├── README.md                                           ← you are here
├── problem_brief.md                                    ← problem spec, dataset, modeling constraints, scoring contract
├── requirements.txt                                    ← pinned, Python 3.12.10 (training environment)
├── requirements-api.txt                                ← lean runtime for the FastAPI service
├── Dockerfile                                          ← container image for the API
├── sba_loan_default_model.ipynb                        ← end-to-end training notebook
├── score_model.py                                      ← production scoring entrypoint
├── Makefile                                            ← test · lint · api · reference · evaluate · golden · docker-smoke
├── api/                                                ← FastAPI service wrapping score_model
│   ├── main.py                                         ← app + endpoints + lifespan loader + request logging
│   ├── schemas.py                                      ← Pydantic v2 request/response models
│   └── monitoring.py                                   ← PSI drift against the training reference
├── scripts/
│   ├── evaluate.py                                     ← held-out calibration / slices / cost-weighted threshold
│   ├── build_reference.py                              ← freeze the training feature distribution for drift
│   └── freeze_golden.py                                ← re-freeze the golden-prediction test fixture
├── tests/
│   ├── test_api.py                                     ← HTTP contract
│   ├── test_contract.py                                ← golden predictions, model-card pins, adversarial inputs, drift
│   ├── golden_predictions.json · example_request.json
├── .github/workflows/ci.yml                            ← ruff + pytest + Docker smoke on push/PR
├── artifacts/                                          ← models + encoders + evaluation + diagnostic plots
│   ├── xgb_model_1.json                                ← shipped model (weight 1.0)
│   ├── xgb_model_2.json                                ← weight 0.0, kept for audit
│   ├── ensemble_metadata.json                          ← weights, threshold, feature order
│   ├── eval_report.json                                ← held-out metrics, calibration, slices, threshold sweep
│   ├── reference_stats.json                            ← drift reference (histograms only, no loan rows)
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

## FastAPI service

The same scoring pipeline is exposed as a production-shaped HTTP service. The Gradio
Space is for human exploration; the FastAPI app is the integration surface
(decision-support widget, internal review tool, batch scoring job, etc.).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health`        | Liveness + readiness; `model_loaded`, drift reference loaded, uptime |
| `GET`  | `/model-info`    | Honest model card: features, threshold, AUCPR, **actual** ensemble weights, held-out calibration (ECE), worst slice, cost-optimal threshold |
| `POST` | `/predict`       | Score one loan record |
| `POST` | `/predict-batch` | Score up to 10,000 records in one round-trip, order preserved; batches of 200+ rows get a PSI drift summary |
| `POST` | `/drift`         | PSI of a batch's engineered features and score distribution against the training reference |
| `GET`  | `/docs`          | Swagger UI (auto-generated) |
| `GET`  | `/openapi.json`  | OpenAPI 3.1 schema |

Input validation is enforced by Pydantic v2 with `extra='forbid'`: unknown fields
return `422`, not a silent drop. Negative `GrAppv`, out-of-range `UrbanRural`, etc.
are caught at the boundary, not by the model. Every response carries `X-Request-ID`
and `X-Process-Time-Ms`, and each scoring request writes one JSON log line.

**Drift monitoring** ([`api/monitoring.py`](api/monitoring.py)): `scripts/build_reference.py`
freezes 10-bin histograms of the 27 numeric engineered features and of the model's own
score on the 679k train+validation rows (test rows excluded) into
`artifacts/reference_stats.json`. The API computes the Population Stability Index of an
incoming batch against it and reports stable (< 0.10), moderate (0.10-0.25) or shifted
(> 0.25) per feature, plus the score-distribution PSI and predicted default rate. It is
the "monthly PSI" the earlier README asked for, wired into the service instead of
described.

### Run locally

```bash
# From the project root, in a 3.12 venv
pip install -r requirements-api.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000/docs for Swagger
```

### Run in Docker

```bash
docker build -t sba-api .
docker run --rm -p 8000:8000 sba-api
# http://localhost:8000/       the scoring UI (same Gradio app as the Hugging Face Space, served at /ui)
# http://localhost:8000/docs   the interactive API docs
curl http://localhost:8000/health
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "index": 1, "City": "GARDENA", "State": "CA", "Zip": 90249,
    "Bank": "ONEUNITED BANK", "BankState": "MA", "NAICS": 0,
    "NoEmp": 6, "NewExist": 2.0, "CreateJob": 0, "RetainedJob": 0,
    "FranchiseCode": 1, "UrbanRural": 0, "RevLineCr": "0", "LowDoc": "Y",
    "DisbursementGross": 50000.0, "BalanceGross": 0.0,
    "GrAppv": 50000.0, "SBA_Appv": 40000.0
  }'
```

```json
{
  "predictions": [
    {"index": 1, "label": 0, "probability_0": 0.78, "probability_1": 0.22}
  ]
}
```

### Tests

```bash
make test        # 26 tests, no network
```

- `tests/test_api.py`: the HTTP contract (health, model card, single and batch
  scoring, field validation).
- `tests/test_contract.py`: **golden predictions** (eight holdout rows must reproduce
  the frozen probabilities to 1e-4, so any silent change to feature engineering,
  encoders or the booster fails the build), **model-card pins** (the README's numbers
  are read back from the artifacts), **adversarial inputs** (unseen states, dirty
  flags, coded-missing values, extreme magnitudes, numeric strings, oversized batches),
  and **drift** (PSI is ~0 on an in-distribution sample and flags `log_disbursement`
  and `GrAppv` as shifted when every loan is scaled 50x).

`make golden` re-freezes the golden file after a deliberate retrain; `make evaluate`
and `make reference` rebuild the evaluation report and drift reference from the
labelled archive (gitignored, 87 MB).

### Design notes

- Artifacts load **once per process** (`score_model.load_artifacts` is cached) and are
  touched at startup via FastAPI's `lifespan` context. Before the cache, every call
  re-read the 49 MB booster from disk: 3.6 s per request. After: 23 ms for one loan,
  0.4 s for 10,000.
- A synthetic warm-up row runs through the full pipeline at boot, so the encoder
  state and XGBoost JIT path are exercised before traffic hits.
- CORS is open (`*`) because this is a portfolio demo. A real deployment would
  lock it down to the calling origin.
- One Uvicorn worker is correct for a CPU-bound XGBoost service; scale horizontally
  (more containers) rather than more workers per container.

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

The Hugging Face Space ([bass990/SBA-Loan-Default-Prediction](https://huggingface.co/spaces/bass990/SBA-Loan-Default-Prediction))
wraps the same `score_model.py` the API and the golden-prediction tests use, so the three
cannot drift apart:

1. Upload a CSV in the project schema (same columns as training, no `MIS_Status`), or press
   **Load 500-row sample**.
2. Pick the decision threshold: **F1-optimal 0.66** (the model-card number) or
   **cost-optimal 0.35** (lowest expected loss per loan under the cost model in `scripts/evaluate.py`).
3. Click **Score**. The summary states how many records were flagged, the expected-loss
   comparison between the two thresholds, and a calibration warning (ECE 0.21: the raw
   probabilities over-state default risk, use them to rank and threshold).
4. The results table has three columns: `index`, `predicted_default`, `default_probability` (3 dp).
5. **Download results CSV** persists the scored rows.

The same UI is served by the Docker image at `/` (API docs at `/docs`). Publishing a new
version is `hf auth login` once, then `python scripts/publish_space.py`; deployment notes are
in `huggingface-spaces/deployment_notes.md`.

---

## What I'd want before deploying this for real

This is a portfolio model, not an SBA-approved underwriting model. If a loan officer
were going to use it on a live application tomorrow, I would want:

1. **Drift monitoring: built.** `/drift` and the batch drift summary compute PSI
   against the frozen training distribution. What is still missing is the
   operational half: a scheduler that scores the live application stream monthly,
   stores the PSI series, and pages someone when a feature crosses 0.25. Training
   data ends in 2014; NAICS mixes, `LowDoc` rules and guarantee fractions have all
   moved since.
2. **Slice analysis: measured, not yet enforced.** The per-state, per-sector and
   per-loan-size tables above are in `eval_report.json`. Turning them into service
   minima (for example, "false-negative rate on loans above $500k must be below 60%")
   would fail the current model, which is the point of writing them down.
3. **Cost-weighted threshold: measured, not yet shipped.** The loss sweep shows a
   0.35 threshold cutting expected loss by 22% under illustrative assumptions.
   Shipping it needs the real loss-given-default and margin figures from the
   lender, and per-segment thresholds rather than one global number.
4. **Calibration: measured, not yet fixed.** ECE 0.21 means the raw probability
   over-states default risk roughly two- to three-fold in the middle of the range.
   Isotonic or Platt calibration fitted on the validation split, with the
   reliability table re-checked on test, is the next modelling change.
5. **Adverse-action transparency.** ECOA requires lenders to give applicants
   specific reasons for denial. The SHAP waterfall plots are a starting point, but
   the production system would need a curated reason-code dictionary that maps
   SHAP attributions to legally compliant language, not the raw feature names.

---

## What went wrong along the way

**The ensemble that was one model.** The weight search over the two XGBoost configurations settled on `(1.0, 0.0)`. Model 2 added nothing, and its config (a different `scale_pos_weight` and `min_child_weight` regime) traded recall for precision in a region the F1 threshold was never going to favour. I kept the search, the plot and the metadata exactly as they came out rather than describing an ensemble, and the scoring UI no longer prints ensemble weights.

**Measuring the better threshold and then not using it.** `scripts/evaluate.py` shows that scoring at 0.35 instead of the F1-optimal 0.66 lowers expected loss per loan from $5,330 to $4,140 under the stated cost assumptions. The Hugging Face app kept using 0.66 and did not mention the alternative. It now offers both by name, with the expected-loss figures next to them.

**Probabilities being read as probabilities.** Expected calibration error is 0.21. In the middle of the range the model's output is two to three times the observed default rate. Nothing in the original app said so, and a loan officer reading 0.66 as a 66% chance of default would be wrong. Every scored result now carries that warning. Isotonic calibration on the validation split is the next modelling change.

**Two copies of the scoring code.** The Space had its own cleaning and feature-engineering functions, separate from `score_model.py`, the module the API and the golden-prediction tests protect. They agreed when I diffed them, but nothing kept them that way. The Space imports the same module now and the publish script uploads it with the app.

**Python 3.14.** My machine's system Python is 3.14 with a NumPy build that segfaults on import; the first attempt to write the sample CSV died with a stack trace before pandas loaded. The project pins 3.12 in its own virtual environment and CI retrains and diffs the model on 3.12.

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
