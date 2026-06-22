#!/usr/bin/env python3
"""Evaluate EMBER2024 LightGBM on the user's existing test split."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
import thrember

from common import default_output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=default_output_root() / "ember2024")
    parser.add_argument("--task", choices=["binary", "family"], default="binary")
    args = parser.parse_args()
    output = args.output_root.expanduser() / args.task
    X_test, y_test = thrember.read_vectorized_features(output, "test")
    model = lgb.Booster(model_file=str(output / "model.txt"))
    probabilities = model.predict(X_test)
    if args.task == "binary":
        predictions = (probabilities >= 0.5).astype(int)
        results = {
            "samples": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
    else:
        predictions = np.argmax(probabilities, axis=1)
        results = {
            "samples": int(len(y_test)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        }
    (output / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

