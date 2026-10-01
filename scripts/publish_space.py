"""Publish the Gradio scoring app to the Hugging Face Space.

    huggingface-cli login            # once, with a token that has write access to the Space
    python scripts/publish_space.py  # uploads app.py, score_model.py, sample_500.csv, requirements.txt, artifacts/

Uploads exactly the files the Space needs (see huggingface-spaces/deployment_notes.md)
in one commit, then prints the Space URL. Nothing else in the Space repo is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE = "bass990/SBA-Loan-Default-Prediction"
FILES = {
    "app.py": ROOT / "huggingface-spaces" / "app.py",
    "score_model.py": ROOT / "score_model.py",
    "sample_500.csv": ROOT / "huggingface-spaces" / "sample_500.csv",
    "requirements.txt": ROOT / "huggingface-spaces" / "requirements.txt",
    **{f"artifacts/{name}": ROOT / "artifacts" / name for name in (
        "xgb_model_1.json", "xgb_model_2.json", "label_encoders.pkl", "median_fill.pkl",
        "ensemble_metadata.json", "eval_report.json")},
}


def main() -> int:
    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        print(f"not logged in to Hugging Face ({str(exc)[:80]}). Run: huggingface-cli login")
        return 1
    missing = [p for p in FILES.values() if not p.exists()]
    if missing:
        print("missing:", *missing, sep="\n  ")
        return 1
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in FILES.items()]
    info = api.create_commit(repo_id=SPACE, repo_type="space", operations=ops,
                             commit_message="Scoring UI v2: shared score_model, threshold choice, calibration note, sample CSV")
    print(f"published as {user}: {info.commit_url}\nhttps://huggingface.co/spaces/{SPACE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
