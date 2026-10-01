"""Input-drift monitoring for the scoring service.

Population Stability Index (PSI) per engineered feature, computed against a
reference distribution frozen from the training data
(`artifacts/reference_stats.json`, built by `scripts/build_reference.py`).

PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%).
Conventional reading: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 the
feature distribution has moved enough that the model's learned relationships
may no longer hold. The thresholds are a rule of thumb, not a statistical
test, which is why the endpoint reports the per-feature values and lets the
caller decide, rather than emitting a single pass/fail.

The batch endpoint attaches a drift summary to any request with at least
`MIN_ROWS_FOR_DRIFT` rows; smaller batches are too noisy to bin.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ROWS_FOR_DRIFT = 200
_EPS = 1e-6
STABLE, MODERATE = 0.10, 0.25


def load_reference(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _hist_share(values: np.ndarray, edges: list[float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=np.asarray(edges, dtype=float))
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return counts / total


def psi(expected_share: np.ndarray, actual_share: np.ndarray) -> float:
    e = np.clip(expected_share, _EPS, None)
    a = np.clip(actual_share, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def _finite(x):
    """JSON cannot carry NaN; a missing reference statistic is None."""
    try:
        return None if x is None or not np.isfinite(float(x)) else float(x)
    except (TypeError, ValueError):
        return None


def _status(value: float) -> str:
    if value < STABLE:
        return "stable"
    if value < MODERATE:
        return "moderate"
    return "shifted"


def drift_report(features: pd.DataFrame, reference: dict, probabilities: np.ndarray | None = None) -> dict:
    """Per-feature PSI of `features` (engineered, pre-encoding) vs the reference."""
    per_feature: dict[str, dict] = {}
    for name, ref in reference["features"].items():
        if name not in features.columns:
            continue
        col = pd.to_numeric(features[name], errors="coerce").dropna().to_numpy(dtype=float)
        if col.size == 0:
            continue
        share = _hist_share(col, ref["edges"])
        value = psi(np.asarray(ref["share"]), share)
        per_feature[name] = {"psi": round(value, 4), "status": _status(value)}
    out = {
        "n_rows": int(len(features)),
        "reference": {"source": reference.get("source"), "n_rows": reference.get("n_rows"), "built_at": reference.get("built_at")},
        "features": per_feature,
        "n_shifted": sum(1 for v in per_feature.values() if v["status"] == "shifted"),
        "n_moderate": sum(1 for v in per_feature.values() if v["status"] == "moderate"),
        "worst": sorted(per_feature.items(), key=lambda kv: -kv[1]["psi"])[:5],
    }
    if probabilities is not None and "score" in reference:
        share = _hist_share(np.asarray(probabilities, dtype=float), reference["score"]["edges"])
        value = psi(np.asarray(reference["score"]["share"]), share)
        out["score_psi"] = {"psi": round(value, 4), "status": _status(value),
                            "predicted_default_rate": round(float(np.mean(np.asarray(probabilities) >= reference["score"]["threshold"])), 4),
                            "reference_default_rate": _finite(reference["score"].get("label_rate"))}
    return out
