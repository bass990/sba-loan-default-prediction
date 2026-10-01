"""Re-freeze tests/golden_predictions.json from the shipped artifacts.

Run after a deliberate retrain; the golden test then guards against
accidental changes to feature engineering, encoders or the booster.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import score_model as sm  # noqa: E402

HOLDOUT = ROOT / "data" / "SBA_loans_project_2_holdout_students_valid.csv"
OUT = ROOT / "tests" / "golden_predictions.json"


def main() -> int:
    df = pd.read_csv(HOLDOUT).head(8)
    out = sm.score(df)
    rows = []
    for (_, i), (_, o) in zip(df.iterrows(), out.iterrows(), strict=True):
        rec = {k: (v.item() if hasattr(v, "item") else v) for k, v in i.items()}
        rows.append({"input": rec, "label": int(o.label), "probability_1": round(float(o.probability_1), 6)})
    OUT.write_text(json.dumps({"note": "First 8 rows of the holdout schema file scored by the shipped artifacts; "
                                       "regenerate with make golden after retraining.", "rows": rows}, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
