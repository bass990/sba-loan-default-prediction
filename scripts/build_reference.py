"""Freeze the training-data feature distribution for drift monitoring.

    python scripts/build_reference.py [--csv data/SBA_loans_project_2.csv | --zip data/SBA_loans_project_2.zip]

Writes artifacts/reference_stats.json: for each of the 31 engineered features,
10 quantile-based bin edges and the share of training rows per bin; plus the
distribution of the model's own probability_1 on the training rows and the
label prevalence. The API's /drift endpoint and /predict-batch drift summary
compare incoming batches against this file with PSI.

Only aggregate histograms leave this script; no loan-level rows are written.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import score_model as sm  # noqa: E402

N_BINS = 10


def _edges(values: np.ndarray) -> list[float]:
    qs = np.quantile(values, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(qs)
    if len(edges) < 2:  # constant feature
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    edges[0], edges[-1] = -np.inf, np.inf
    return [float(e) for e in edges]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--zip", default=str(ROOT / "data" / "SBA_loans_project_2.zip"))
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "reference_stats.json"))
    ap.add_argument("--sample", type=int, default=0, help="optional row cap for a quick build")
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, low_memory=False)
        source = Path(args.csv).name
    else:
        with zipfile.ZipFile(args.zip) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            with z.open(name) as f:
                df = pd.read_csv(f, low_memory=False)
        source = name
    if args.sample:
        df = df.sample(args.sample, random_state=0)
    elif "MIS_Status" in df.columns and not args.csv:
        # Reproduce the notebook's stratified 15% test split (SEED 42) and keep
        # only train + validation rows, so the reference never sees the test set.
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
        keep, _test = train_test_split(df, test_size=0.15, random_state=42, stratify=df["MIS_Status"])
        df = keep
        source = f"{source} (train+val split, test rows excluded)"
    if "index" not in df.columns:
        df.insert(0, "index", range(len(df)))
    labels = None
    if "MIS_Status" in df.columns:
        raw = df["MIS_Status"]
        labels = pd.to_numeric(raw, errors="coerce") if pd.api.types.is_numeric_dtype(raw) else \
            raw.astype(str).str.strip().str.upper().map({"P I F": 0, "CHGOFF": 1})
    print(f"{source}: {len(df):,} rows")

    art = sm.load_artifacts()
    feats = sm._engineer_features(sm._clean_df(df.drop(columns=["MIS_Status"], errors="ignore")))
    ref_features = {}
    for name in art["metadata"]["feature_names"]:
        col = pd.to_numeric(feats[name], errors="coerce").dropna().to_numpy(dtype=float) if name in feats.columns else np.array([])
        if col.size == 0:
            continue
        edges = _edges(col)
        counts, _ = np.histogram(col, bins=np.asarray(edges))
        ref_features[name] = {"edges": edges, "share": [float(c) / counts.sum() for c in counts]}

    probs = sm.score(df.drop(columns=["MIS_Status"], errors="ignore"))["probability_1"].to_numpy()
    s_edges = [-np.inf, *np.linspace(0.1, 0.9, 9).tolist(), np.inf]
    s_counts, _ = np.histogram(probs, bins=np.asarray(s_edges))
    out = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "n_rows": int(len(df)),
        "features": ref_features,
        "score": {"edges": [float(e) for e in s_edges], "share": [float(c) / s_counts.sum() for c in s_counts],
                  "threshold": art["metadata"]["threshold_ensemble"],
                  "label_rate": round(float(labels.mean()), 4) if labels is not None else None,
                  "predicted_default_rate": round(float((probs >= art["metadata"]["threshold_ensemble"]).mean()), 4)},
    }
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {len(ref_features)} features, label rate {out['score']['label_rate']}, "
          f"predicted default rate {out['score']['predicted_default_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
