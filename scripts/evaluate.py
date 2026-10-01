"""Held-out evaluation of the shipped model: calibration, slices, cost-weighted threshold.

    python scripts/evaluate.py [--zip data/SBA_loans_project_2.zip] [--lgd 0.5 --margin 0.05]

Reproduces the notebook's stratified 15% test split (SEED 42) from the labelled
training archive, scores it with the production `score_model.score()`, and
writes artifacts/eval_report.json with:

  * headline: AUCPR, ROC AUC, precision/recall/F1 at the shipped 0.66 threshold
    (AUCPR must reproduce the model card's 0.5665; the test asserts it);
  * calibration: Brier score, expected calibration error, a 10-bin reliability
    table (the README's "what I'd want" item 4);
  * slices: AUCPR, recall and false-negative rate by state (top 10 by volume),
    NAICS sector and loan-size bucket (item 2);
  * cost-weighted threshold: expected loss per loan under a loss-given-default
    and interest-margin assumption, swept over thresholds (item 3). The
    assumptions are parameters, not claims; the point is to show the operating
    point moves when the loss is asymmetric.

Everything here is on the held-out test rows only; nothing is refit.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import score_model as sm  # noqa: E402


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, acc, n = float(p[m].mean()), float(y[m].mean()), int(m.sum())
        ece += n / len(p) * abs(acc - conf)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": n, "mean_predicted": round(conf, 4), "observed_default_rate": round(acc, 4)})
    return round(float(ece), 4), table


def _slice_metrics(df: pd.DataFrame, y: np.ndarray, p: np.ndarray, thr: float, key: str, top: int | None = None) -> list[dict]:
    rows = []
    groups = df[key].value_counts()
    keys = groups.index[:top] if top else groups.index
    for k in keys:
        m = (df[key] == k).to_numpy()
        yy, pp = y[m], p[m]
        if yy.sum() < 20 or (1 - yy).sum() < 20:
            continue
        pred = (pp >= thr).astype(int)
        rows.append({key: str(k), "n": int(m.sum()), "default_rate": round(float(yy.mean()), 4),
                     "aucpr": round(float(average_precision_score(yy, pp)), 4),
                     "recall": round(float(recall_score(yy, pred)), 4),
                     "false_negative_rate": round(float(1 - recall_score(yy, pred)), 4),
                     "precision": round(float(precision_score(yy, pred, zero_division=0)), 4)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "data" / "SBA_loans_project_2.zip"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eval_report.json"))
    ap.add_argument("--lgd", type=float, default=0.5, help="loss given default as a share of the SBA-guaranteed amount")
    ap.add_argument("--margin", type=float, default=0.05, help="interest margin earned on a loan that repays, share of GrAppv")
    args = ap.parse_args()

    with zipfile.ZipFile(args.zip) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, low_memory=False)
    y_all = df["MIS_Status"]
    _, test = train_test_split(df, test_size=0.15, random_state=42, stratify=y_all)
    test = test.reset_index(drop=True)
    y = test["MIS_Status"].astype(str).str.strip().str.upper().map({"P I F": 0, "CHGOFF": 1}).to_numpy()
    if np.isnan(y.astype(float)).any():  # already 0/1 encoded in some exports
        y = pd.to_numeric(test["MIS_Status"], errors="coerce").fillna(0).astype(int).to_numpy()
    X = test.drop(columns=["MIS_Status"])
    if "index" not in X.columns:
        X.insert(0, "index", range(len(X)))
    print(f"test split: {len(test):,} rows, default rate {y.mean():.4f}")

    art = sm.load_artifacts()
    thr = float(art["metadata"]["threshold_ensemble"])
    p = sm.score(X)["probability_1"].to_numpy()
    pred = (p >= thr).astype(int)

    headline = {
        "n_test": int(len(y)), "default_rate": round(float(y.mean()), 4), "threshold": round(thr, 4),
        "aucpr": round(float(average_precision_score(y, p)), 4), "aucpr_model_card": round(float(art["metadata"]["aucpr_test_ens"]), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "precision": round(float(precision_score(y, pred)), 4), "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4), "predicted_default_rate": round(float(pred.mean()), 4),
    }
    ece, table = _ece(y, p)
    calibration = {"brier": round(float(brier_score_loss(y, p)), 4), "ece": ece, "reliability": table,
                   "note": "scale_pos_weight=4.71 inflates predicted probabilities; ranking is good, the raw number over-states default risk (see reliability table)"}

    feats = sm._engineer_features(sm._clean_df(X))
    feats["loan_size_bucket"] = pd.cut(feats["GrAppv"], [0, 50_000, 150_000, 500_000, np.inf], labels=["<=50k", "50-150k", "150-500k", ">500k"])
    slices = {
        "state_top10": _slice_metrics(feats, y, p, thr, "State", top=10),
        "naics_sector": _slice_metrics(feats, y, p, thr, "naics_sector"),
        "loan_size": _slice_metrics(feats, y, p, thr, "loan_size_bucket"),
    }
    worst_fnr = max((r for s in slices.values() for r in s), key=lambda r: r["false_negative_rate"])

    # cost-weighted threshold: approve iff p < t. Loss from approved defaults + foregone margin from denied repayers.
    exposure = feats["SBA_Appv"].to_numpy(dtype=float)
    gr = feats["GrAppv"].to_numpy(dtype=float)
    sweep = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        approve = p < t
        loss = float((exposure[approve & (y == 1)] * args.lgd).sum() + (gr[~approve & (y == 0)] * args.margin).sum())
        sweep.append({"threshold": float(t), "expected_loss_per_loan": round(loss / len(y), 2), "approval_rate": round(float(approve.mean()), 4),
                      "recall": round(float(recall_score(y, (p >= t).astype(int))), 4)})
    best = min(sweep, key=lambda r: r["expected_loss_per_loan"])
    at_f1 = next(r for r in sweep if abs(r["threshold"] - round(thr, 2)) < 0.03) if any(abs(r["threshold"] - round(thr, 2)) < 0.03 for r in sweep) else None
    cost = {"assumptions": {"lgd_share_of_sba_appv": args.lgd, "margin_share_of_gr_appv": args.margin},
            "best_threshold": best, "f1_threshold": at_f1, "sweep": sweep,
            "note": "illustrative loss model; the assumptions are inputs, not measured SBA economics"}

    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "source": f"{name} stratified 15% test split, seed 42",
              "headline": headline, "calibration": calibration, "slices": slices, "worst_slice_by_fnr": worst_fnr, "cost_weighted_threshold": cost}
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(headline, indent=1))
    print(f"calibration: Brier {calibration['brier']}, ECE {ece}")
    print(f"worst slice by FNR: {worst_fnr}")
    print(f"cost-optimal threshold {best['threshold']} (loss/loan ${best['expected_loss_per_loan']}) vs F1 threshold {thr:.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
